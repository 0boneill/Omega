#include <pio.h>
#include <pio_internal.h>


#define MALLOC_FILL_ARRAY(type, n, fill, arr) \
  arr = malloc(n * sizeof (type));	      \
  if(fill != NULL)                                       \
    for(int _i=0; _i<n; _i++)			\
      ((type *) arr)[_i] = *((type *) fill)



int pio_write_darray_nc(file_desc_t *file, io_desc_t *iodesc, const int vid, void *IOBUF, void *fillvalue)
{
  iosystem_desc_t *ios;
  PIO_Offset *start, *count;
  var_desc_t *vdesc;
  int ndims;
  int ierr;
  int msg;
  int mpierr;
  int i;
  void *tmp_buf=NULL;
  int dsize;
  MPI_Status status;

  ierr = PIO_NOERR;

  ios = file->iosystem;
  if(ios == NULL)
    return PIO_EBADID;

  vdesc = (file->varlist)+vid;

  if(vdesc == NULL)
    return PIO_EBADID;

  ndims = iodesc->ndims;
  
  msg = 0;

  if(ios->async_interface && ! ios->ioproc){
    if(ios->comp_rank==0) 
      mpierr = MPI_Send(&msg, 1,MPI_INT, ios->ioroot, 1, ios->union_comm);
    mpierr = MPI_Bcast(&(file->fh),1, MPI_INT, ios->compmaster, ios->intercomm);
  }

  
  if(ios->ioproc){
    int ndims = iodesc->ndims;
    int ncid = file->fh;
    size_t tstart[ndims], tcount[ndims];

    // this is a time dependent multidimensional array
    if(vdesc->record >= 0 && iodesc->start[1]>=0){
      start = (PIO_Offset *) calloc(ndims+1, sizeof(PIO_Offset));
      count = (PIO_Offset *) calloc(ndims+1, sizeof(PIO_Offset));
      for(int i=0;i<ndims;i++){
	start[i] = iodesc->start[i];
	count[i] = iodesc->count[i];
	start[ndims] = vdesc->record;
	count[ndims] = 1;
      }
      // this is a timedependent single value
    }else if(vdesc->record >= 0){
      start = (PIO_Offset *) malloc( sizeof(PIO_Offset));
      count = (PIO_Offset *) malloc( sizeof(PIO_Offset));
      start[0] = vdesc->record;
      count[0] = 1;
      // Non-time dependent array
    }else{
      //start = (PIO_Offset *) calloc(ndims, sizeof(PIO_Offset));
      //count = (PIO_Offset *) calloc(ndims, sizeof(PIO_Offset));
      start  = iodesc->start;
      count = iodesc->count;
    }      


    switch(file->iotype){
#ifdef _NETCDF
#ifdef _NETCDF4
    case PIO_IOTYPE_NETCDF4P:
      ierr = nc_var_par_access(ncid, vid, NC_COLLECTIVE);
      switch(iodesc->basetype){
      case MPI_DOUBLE:
      case MPI_REAL8:
	ierr = nc_put_vara_double (ncid, vid,(size_t *) start,(size_t *) count, (const double *) IOBUF); 
	break;
      case MPI_INTEGER:
	ierr = nc_put_vara_int (ncid, vid, (size_t *) start, (size_t *) count, (const int *) IOBUF); 
	break;
      case MPI_FLOAT:
      case MPI_REAL4:
	ierr = nc_put_vara_float (ncid, vid, (size_t *) start, (size_t *) count, (const float *) IOBUF); 
	break;
    default:
      fprintf(stderr,"Type not recognized %d in pioc_write_darray\n",(int) iodesc->basetype);
    }
    case PIO_IOTYPE_NETCDF4C:
#endif
    case PIO_IOTYPE_NETCDF:
      mpierr = MPI_Type_size(iodesc->basetype, &dsize);

      if(ios->io_rank==0){
	for(i=0;i<ios->num_aiotasks;i++){
	  if(i==0){	    
	    for(int j=0;j<ndims;j++){
	      tstart[j] =  start[j];
	      tcount[j] =  count[j];
	      tmp_buf = IOBUF;
	    }
	  }else{
	    if(i==1){
	      tmp_buf = malloc(iodesc->maxiobuflen * dsize);	
	    }
	    
	    mpierr = MPI_Send( &ierr, 1, MPI_INT, i, 0, ios->io_comm);  // handshake - tell the sending task I'm ready
	    mpierr = MPI_Recv( tstart, ndims, MPI_INT, i, ios->num_iotasks+i, ios->io_comm, &status);
	    mpierr = MPI_Recv( tcount, ndims, MPI_INT, i,2*ios->num_iotasks+i, ios->io_comm, &status);
	    mpierr = MPI_Recv( tmp_buf, iodesc->maxiobuflen, iodesc->basetype, i, i, ios->io_comm, &status);
	  }
	  if(iodesc->basetype == MPI_INTEGER){
	    ierr = nc_put_vara_int (ncid, vid, tstart, tcount, (const int *) tmp_buf); 
	  }else if(iodesc->basetype == MPI_DOUBLE || iodesc->basetype == MPI_REAL8){
	    ierr = nc_put_vara_double (ncid, vid, tstart, tcount, (const double *) tmp_buf); 
	  }else if(iodesc->basetype == MPI_FLOAT || iodesc->basetype == MPI_REAL4){
	    //    for(int j=0;j<ndims;j++)
	    //  printf("tstart[%d] %ld tcount %ld %ld\n",j,tstart[j],tcount[j], iodesc->maxiobuflen);
	    
	    ierr = nc_put_vara_float (ncid,vid, tstart, tcount, (const float *) tmp_buf); 
	  }else{
	    fprintf(stderr,"Type not recognized %d in pioc_write_darray\n",(int) iodesc->basetype);
	  }
	}     
      }else if(count[0]>0){
	for(i=0;i<ndims;i++){
	  tstart[i] = (size_t) start[i];
	  tcount[i] = (size_t) count[i];
	}
	mpierr = MPI_Recv( &ierr, 1, MPI_INT, 0, 0, ios->io_comm, &status);  // task0 is ready to recieve
	mpierr = MPI_Rsend( tstart, ndims, MPI_INT, 0, ios->num_iotasks+ios->io_rank, ios->io_comm);
	mpierr = MPI_Rsend( tcount, ndims, MPI_INT, 0,2*ios->num_iotasks+ios->io_rank, ios->io_comm);
	mpierr = MPI_Rsend( IOBUF, iodesc->maxiobuflen, iodesc->basetype, 0, ios->io_rank, ios->io_comm);
      }
      break;
#endif
#ifdef _PNETCDF
    case PIO_IOTYPE_PNETCDF:
      
      for( i=0,dsize=1;i<ndims;i++)
	dsize*=count[i];

      //      printf("pnet %ld %ld %ld\n",start[0],count[0],dsize);
      // for(i=0;i<dsize;i++)
      //	printf("iobuf(%d) %d\n",i,((int *)IOBUF)[i]);

	ierr = ncmpi_iput_vara(ncid, vid,  start, count, IOBUF,
			       dsize, iodesc->basetype, &(vdesc->request));
      break;
#endif
    default:
      ierr = iotype_error(file->iotype,__FILE__,__LINE__);
    }
  }
  
  ierr = check_netcdf(file, ierr, __FILE__,__LINE__);
  if(tmp_buf != NULL && tmp_buf != IOBUF)
    free(tmp_buf);



  return ierr;
}

int PIOc_write_darray(const int ncid, const int vid, const int ioid, const PIO_Offset arraylen, void *array, void *fillvalue)
{
  iosystem_desc_t *ios;
  file_desc_t *file;
  io_desc_t *iodesc;
  void *iobuf;
  size_t vsize, rlen;
  int ierr;
  MPI_Datatype vtype;

  ierr = PIO_NOERR;

  file = pio_get_file_from_id(ncid);
  if(file == NULL)
    return PIO_EBADID;

  iodesc = pio_get_iodesc_from_id(ioid);
  if(iodesc == NULL)
    return PIO_EBADID;

  iobuf = NULL;

  ios = file->iosystem;

  if(ios->ioproc){
    rlen = iodesc->llen;
    vtype = (MPI_Datatype) iodesc->basetype;
    if(vtype == MPI_INTEGER){
      MALLOC_FILL_ARRAY(int, rlen, fillvalue, iobuf);
    }else if(vtype == MPI_FLOAT || vtype == MPI_REAL4){
      MALLOC_FILL_ARRAY(float, rlen, fillvalue, iobuf);
    }else if(vtype == MPI_DOUBLE || vtype == MPI_REAL8){
      MALLOC_FILL_ARRAY(double, rlen, fillvalue, iobuf);
    }else if(vtype == MPI_CHARACTER){
      MALLOC_FILL_ARRAY(char, rlen, fillvalue, iobuf);
    }else{
      fprintf(stderr,"Type not recognized %d in pioc_write_darray\n",vtype);
    }
  }


  ierr = box_rearrange_comp2io(ios, iodesc, array, iobuf, 0, 0);

  switch(file->iotype){
  case PIO_IOTYPE_PBINARY:
    break;
  case PIO_IOTYPE_PNETCDF:
  case PIO_IOTYPE_NETCDF:
  case PIO_IOTYPE_NETCDF4P:
  case PIO_IOTYPE_NETCDF4C:
    ierr = pio_write_darray_nc(file, iodesc, vid, iobuf, fillvalue);
  }

  if(file->iotype != PIO_IOTYPE_PNETCDF && iobuf != NULL)
    free(iobuf);

  return ierr;

}

int pio_read_darray_nc(file_desc_t *file, io_desc_t *iodesc, const int vid, void *IOBUF)
{
  int ierr=PIO_NOERR;
  iosystem_desc_t *ios;
  var_desc_t *vdesc;
  int ndims;
  MPI_Status status;
  int i;

  ios = file->iosystem;
  if(ios == NULL)
    return PIO_EBADID;

  vdesc = (file->varlist)+vid;

  if(vdesc == NULL)
    return PIO_EBADID;

  ndims = iodesc->ndims;
  if(ios->ioproc){
    size_t start[ndims+1];
    size_t count[ndims+1];
    size_t tmp_start[ndims+1];
    size_t tmp_count[ndims+1];
    size_t tmp_bufsize=1;
    // Non-time dependent array
    for(i=0;i<ndims;i++){
      start[i] = iodesc->start[i];
      count[i] = iodesc->count[i];
    }
    if(vdesc->record >= 0){
      if(iodesc->start[1]>=0){
	// this is a time dependent multidimensional array
	start[ndims] = vdesc->record;
	count[ndims] = 1;
      }else{
	// this is a timedependent single value
	start[0] = vdesc->record;
	count[0] = 1;
      }
    }
    switch(file->iotype){
#ifdef _NETCDF
#ifdef _NETCDF4
    case PIO_IOTYPE_NETCDF4P:
      switch(iodesc->basetype){
      case MPI_DOUBLE:
      case MPI_REAL8:
	ierr = nc_get_vara_double (file->fh, vid,start,count, IOBUF); 
	break;
      case MPI_INTEGER:
	ierr = nc_get_vara_int (file->fh, vid, start, count,  IOBUF); 
	break;
      case MPI_FLOAT:
      case MPI_REAL4:
	ierr = nc_get_vara_float (file->fh, vid, start,  count,  IOBUF); 
	break;
    default:
      fprintf(stderr,"Type not recognized %d in pioc_write_darray\n",(int) iodesc->basetype);
    }	
      break;
    case PIO_IOTYPE_NETCDF4C:
#endif
    case PIO_IOTYPE_NETCDF:
      if(ios->io_rank>0){
	for( i=0;i<ndims; i++){
	  tmp_start[i] = start[i];
	  tmp_count[i] = count[i];
	  tmp_bufsize *= count[i];
	}
	MPI_Send( tmp_count, ndims, MPI_UNSIGNED_LONG, 0, ios->io_rank, ios->io_comm);
	if(tmp_bufsize > 0){
	  MPI_Send( tmp_start, ndims, MPI_UNSIGNED_LONG, 0, ios->io_rank, ios->io_comm);
	  MPI_Recv( IOBUF, tmp_bufsize, iodesc->basetype, 0, ios->io_rank, ios->io_comm, &status);
	}
      }else if(ios->io_rank==0){
	for( i=ios->num_iotasks-1;i>=0;i--){
	  if(i==0){
	    for(int k=0;k<ndims;k++)
	      tmp_count[k] = count[k];
	  }else{
	    MPI_Recv(tmp_count, ndims, MPI_UNSIGNED_LONG, i, i, ios->io_comm, &status);
	  }
	  tmp_bufsize=1;
	  for(int j=0;j<ndims; j++){
	    tmp_bufsize *= tmp_count[j];
	  }
	  if(tmp_bufsize>0){
	    if(i==0){
	      for(int k=0;k<ndims;k++)
		tmp_start[k] = start[k];
	    }else{
	      MPI_Recv(tmp_start, ndims, MPI_UNSIGNED_LONG, i, i, ios->io_comm, &status);
	    }
	    if(vdesc->record >= 0){
	      tmp_start[ndims] = vdesc->record;
	      tmp_count[ndims] = 1;
	    }
	    if(iodesc->basetype == MPI_DOUBLE || iodesc->basetype == MPI_REAL8){
	      ierr = nc_get_vara_double (file->fh, vid, tmp_start, tmp_count, IOBUF); 
	    }else if(iodesc->basetype == MPI_INTEGER){
	      ierr = nc_get_vara_int (file->fh, vid, tmp_start, tmp_count,  IOBUF); 	     
	    }else if(iodesc->basetype == MPI_FLOAT || iodesc->basetype == MPI_REAL4){
	      ierr = nc_get_vara_float (file->fh, vid, tmp_start, tmp_count,  IOBUF); 
	    }else{
	      fprintf(stderr,"Type not recognized %d in pioc_write_darray\n",(int) iodesc->basetype);
	    }	
	    if(i>0){
	      MPI_Rsend(IOBUF, tmp_bufsize, iodesc->basetype, i, i, ios->io_comm);
	    }
	  }
	}
	break;
#endif
#ifdef _PNETCDF
      case PIO_IOTYPE_PNETCDF:
	{
	  PIO_Offset tmp_bufsize=1;
	  for(int j=0;j<ndims; j++){
	    tmp_bufsize *= count[j];
	  }
	  ncmpi_get_vara_all(file->fh, vid,(PIO_Offset *) start,(PIO_Offset *) count, IOBUF, tmp_bufsize, iodesc->basetype);
	}
	break;
#endif
      default:
	ierr = iotype_error(file->iotype,__FILE__,__LINE__);
	
      }
    }
  }
  ierr = check_netcdf(file, ierr, __FILE__,__LINE__);

  return ierr;
}

int PIOc_read_darray(const int ncid, const int vid, const int ioid, const PIO_Offset arraylen, void *array)
{
  iosystem_desc_t *ios;
  file_desc_t *file;
  io_desc_t *iodesc;
  void *iobuf=NULL;
  size_t vsize=0, rlen=0;
  int ierr;
  MPI_Datatype vtype;
 
  file = pio_get_file_from_id(ncid);

  if(file == NULL)
    return PIO_EBADID;

  iodesc = pio_get_iodesc_from_id(ioid);
  if(iodesc == NULL)
    return PIO_EBADID;

  ios = file->iosystem;
  
  if(ios->ioproc){
    rlen = iodesc->llen;

    vtype = (MPI_Datatype) iodesc->basetype;
    if(vtype == MPI_INTEGER){
      iobuf = malloc( rlen*sizeof(int));
    }else if(vtype == MPI_FLOAT || vtype == MPI_REAL4){
      iobuf = malloc( rlen*sizeof(float));
    }else if(vtype == MPI_DOUBLE || vtype == MPI_REAL8){
      iobuf = malloc( rlen*sizeof(double));
    }else if(vtype == MPI_CHARACTER){
      iobuf = malloc( rlen*sizeof(char));
    }else{
      fprintf(stderr,"Type not recognized %d in pioc_read_darray\n",vtype);
    }
    if(iobuf == NULL){
      fprintf(stderr,"malloc failed in pioc_read_darray\n");
      return PIO_ENOMEM;
    } 


  }

  switch(file->iotype){
  case PIO_IOTYPE_PBINARY:
    break;
  case PIO_IOTYPE_PNETCDF:
  case PIO_IOTYPE_NETCDF:
  case PIO_IOTYPE_NETCDF4P:
  case PIO_IOTYPE_NETCDF4C:
    ierr = pio_read_darray_nc(file, iodesc, vid, iobuf);
  }
  ierr = box_rearrange_io2comp(ios, iodesc, iobuf, array, 0, 0);

  if(iobuf != NULL)
    free(iobuf);

  return ierr;

}






int flush_ouput_buffer(file_desc_t *file)
{
  var_desc_t *vardesc;
  int ierr=PIO_NOERR;
#ifdef _PNETCDF
  int request_cnt=0;
  MPI_Request requests[PIO_MAX_VARS];
  int status[PIO_MAX_VARS];

  vardesc = file->varlist;

  for(int i=0;i<PIO_MAX_VARS;i++){
    if(vardesc[i].request != MPI_REQUEST_NULL){
      requests[request_cnt] = vardesc[i].request;
      request_cnt++;
    }
  }
  if(request_cnt>0){
    ierr = ncmpi_wait_all(file->fh, request_cnt,  requests, status);
    
    for(int i=0;i<PIO_MAX_VARS;i++){
      if(vardesc[i].request != MPI_REQUEST_NULL){
	free(vardesc[i].buffer);
	vardesc[i].buffer = NULL;
	vardesc[i].request = MPI_REQUEST_NULL;
      }
    }   
  } 
#endif
  return ierr;
}

