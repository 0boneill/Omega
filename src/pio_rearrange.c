/**
 * @file pio_rearrange.c
 * @author Jim Edwards
 * @date 2014
 * @brief Code to map IO to model decomposition
 *
 * 
 * 
 * 
 * @see  http://code.google.com/p/parallelio/
 */
#include <pio.h>
#include <pio_internal.h>
#include <limits.h>
#define DEF_P2P_MAXREQ 64

int tmpioproc=-1;

// Convert a global array index to a global coordinate value
void gindex_to_coord(const int ndims, const PIO_Offset gindex, const PIO_Offset gstride[], PIO_Offset *gcoord)
{
  PIO_Offset tempindex;
  int i;

  tempindex = gindex;
  for(i=0;i<ndims-1;i++){
    gcoord[i] = tempindex/gstride[i];
    tempindex -= gcoord[i]*gstride[i];
  }
  gcoord[ndims-1] = tempindex;

}

// Convert a global coordinate value into a local array index
PIO_Offset coord_to_lindex(const int ndims, const PIO_Offset lcoord[], const PIO_Offset count[])
{
  PIO_Offset lindex=0;
  PIO_Offset stride=1;

  for(int i=ndims-1; i>=0; i--){
    lindex += lcoord[i]*stride;
    stride = stride*count[i];
  }
  return lindex;

}

void compute_maxIObuffersize(MPI_Comm io_comm, io_desc_t *iodesc)
{
  PIO_Offset iosize, totiosize;
  int i;
  io_region *region;

  //  compute the max io buffer size, for conveneance it is the combined size of all regions
  totiosize=0;
  region = iodesc->firstregion;
  while(region != NULL){
    iosize = 0;
    if(region->count[0]>0)
      iosize=1;
      for(i=0;i<iodesc->ndims;i++)
	iosize*=region->count[i];
      totiosize+=iosize;
    region = region->next;
  }
  // Share the max io buffer size with all io tasks
#ifndef _MPISERIAL
  CheckMPIReturn(MPI_Allreduce(MPI_IN_PLACE, &totiosize, 1, MPI_OFFSET, MPI_MAX, io_comm),__FILE__,__LINE__);
#endif
  
  iodesc->maxiobuflen = totiosize;
}


// Expand a region with the given stride. Given an initial region size,
// this simply checks to see whether the next entries in the map are ahead
// by the given stride, and then the ones after that. Once max_size is
// reached or the next entries fail to match, it returns how far it
// expanded.
int expand_region(const int maplen, const PIO_Offset map[], const int region_size,
                  const int stride, const int max_size)
{
  int i, j, expansion;
  int can_expand;
  // Precondition: maplen >= region_size (thus loop runs at least once).

  can_expand = 1;
  
  for (i = 1; i <= max_size; ++i) {
    expansion = i;
    for (j = 0; j < region_size; ++j) {
      //      if(tmpioproc==1)
      //printf("%d %d %d %d %ld %ld\n",i,j,region_size,stride,map[j+i*region_size],map[j]+i*stride);
      if (map[j + i*region_size] != map[j] + i*stride) {
        can_expand = 0;
        break;
      }
    }
    if (!can_expand) break;
  }
  //  printf("expansion %d\n",expansion);
  return expansion;
}

// Set start and count so that they describe the first region in map.
int find_first_region(const int ndims, const int gdims[],
		      const int maplen, const PIO_Offset map[],
		      PIO_Offset start[], PIO_Offset count[])
{
  int i, region_size, max_size;
  PIO_Offset stride[ndims];
  // Preconditions (which might be useful to check/assert):
  //   ndims is > 0
  //   maplen is > 0
  //   all elements of map are inside the bounds specified by gdims

  stride[ndims-1] = 1;
  for(i=ndims-2;i>=0; --i)
    stride[i] = stride[i+1]*gdims[i+1];

  gindex_to_coord(ndims, map[0], stride, start);

  region_size = 1;
  // For each dimension, figure out how far we can expand in that dimension
  // while staying contiguous in the input array.
  //
  // To avoid something complicated involving recursion, calculate
  // the stride necessary to advance in a given dimension, and feed it into
  // the 1D expand_region function.
  for (i = ndims-1; i >= 0; --i) {
    // Can't expand beyond the array edge.
    max_size = gdims[i] - start[i];
    count[i] = expand_region(maplen, map, region_size, stride[i], max_size);
    region_size = region_size * count[i];
  }
  return region_size;
}




int create_mpi_datatypes(const MPI_Datatype basetype,const int msgcnt,const PIO_Offset dlen, const PIO_Offset mindex[],const int mcount[],
			 int *mfrom, MPI_Datatype mtype[])
{
  PIO_Offset bsizeT[msgcnt];
  int pos;
  int ii;
  PIO_Offset i8blocksize;
  int blocksize;
  PIO_Offset *lindex = NULL;
#ifdef _MPISERIAL
  mtype[0] = basetype * blocksize;
#else
  if(mindex != NULL){
    lindex = (PIO_Offset *) malloc(dlen * sizeof(PIO_Offset));
    memcpy(lindex, mindex, (size_t) (dlen*sizeof(PIO_Offset)));
  }
  bsizeT[0]=0;
  mtype[0] = MPI_DATATYPE_NULL;
  pos = 0;
  ii = 0;
  if(msgcnt>0){
    if(mfrom == NULL){
      for(int i=0;i<msgcnt;i++){
	if(mcount[i]>0){
	  bsizeT[ii] = GCDblocksize(mcount[i], lindex+pos);
	  ii++;
	  pos+=mcount[i];
	}
      }
      blocksize = (int) lgcd_array(ii ,bsizeT);
    }else{
      blocksize=1;
    }
    pos = 0;
    for(int i=0;i< msgcnt; i++){
      if(mcount[i]>0){
	int len = mcount[i]/blocksize;
	//		printf("%s %d %d %d %d %d %d %d\n",__FILE__,__LINE__,i,mcount[i],blocksize,len,pos,dlen);
	int displace[len];
	if(blocksize==1){
	  if(mfrom == NULL){
	    for(int j=0;j<len;j++)
	      displace[j] = (int) (lindex[pos+j]);
	  }else{
	    int k=0;
	    for(int j=0;j<dlen;j++)
	      if(mfrom[j]==i)
		displace[k++] = (int) (lindex[j]);
	  }
	    
	}else{
	  //	  displace[0] = lindex[pos]/blocksize;
	  for(int j=0;j<mcount[i];j++)
	    (lindex+pos)[j]++;
	  for(int j=0;j<len;j++){
	    displace[j]= ((lindex+pos)[j*blocksize]-1);
	  }
	}
	CheckMPIReturn(MPI_Type_create_indexed_block(len, blocksize, displace, basetype, mtype+i),__FILE__,__LINE__);
	CheckMPIReturn(MPI_Type_commit(mtype+i), __FILE__,__LINE__);
	pos+=mcount[i];

      }
    }

  }
  
  free(lindex);
#endif
  return PIO_NOERR;

}


int define_iodesc_datatypes(const iosystem_desc_t ios, io_desc_t *iodesc)
{
  int i;
  if(ios.ioproc){
    //    printf("%d IO: %d\n",ios.io_rank,iodesc->nrecvs);
    if(iodesc->rtype==NULL){
      int ntypes = iodesc->nrecvs;
      iodesc->rtype = (MPI_Datatype *) malloc(ntypes * sizeof(MPI_Datatype));
      for(i=0; i<ntypes; i++){
        iodesc->rtype[i] = MPI_DATATYPE_NULL;
      }
      iodesc->num_rtypes = ntypes;
      /*
      for(i=0;i<iodesc->llen;i++)
	printf("%d ", iodesc->rindex[i]);
      printf("\n"); 
      */
      if(iodesc->rearranger==PIO_REARR_SUBSET){
	create_mpi_datatypes(iodesc->basetype, iodesc->nrecvs, iodesc->llen, iodesc->rindex, iodesc->rcount, iodesc->rfrom, iodesc->rtype);
      }else{
	create_mpi_datatypes(iodesc->basetype, iodesc->nrecvs, iodesc->llen, iodesc->rindex, iodesc->rcount, NULL, iodesc->rtype);
      }
#ifndef _MPISERIAL
      /*      {
	MPI_Aint lb;
	MPI_Aint extent;
	for(i=0;i<ntypes;i++){
	  MPI_Type_get_extent(iodesc->rtype[i], &lb, &extent);
	  printf("%s %d %d %d %d \n",__FILE__,__LINE__,i,lb,extent);
	  
	}
	}*/
#endif
    }
  }


  if(iodesc->stype==NULL){
    int ntypes;
    if(iodesc->rearranger==PIO_REARR_SUBSET)
      ntypes = 1;
    else
      ntypes = ios.num_iotasks;


    //  printf("COMP: %d\n",ntypes);


    iodesc->stype = (MPI_Datatype *) malloc(ntypes * sizeof(MPI_Datatype));
    for(i=0; i<ntypes; i++){
      iodesc->stype[i] = MPI_DATATYPE_NULL;
    }
    iodesc->num_stypes = ntypes;

    create_mpi_datatypes(iodesc->basetype, ntypes, iodesc->ndof, iodesc->sindex, iodesc->scount, NULL, iodesc->stype);
#ifndef _MPISERIAL
    /*    {
      MPI_Aint lb;
      MPI_Aint extent;
      for(i=0;i<ntypes;i++){
	MPI_Type_get_extent(iodesc->stype[i], &lb, &extent);
	printf("%s %d %d %d %d \n",__FILE__,__LINE__,i,lb,extent);
      }
      }*/
#endif
  }

  return PIO_NOERR;

}



/*
int compute_counts(const iosystem_desc_t ios, io_desc_t *iodesc, const int maplen, 
		   const int dest_ioproc[], const PIO_Offset dest_ioindex[], MPI_Comm mycomm)
{

  int i;
  int iorank;

  int rank;
  int ntasks;

  MPI_Comm_rank(mycomm, &rank);
  MPI_Comm_size(mycomm, &ntasks);

  MPI_Datatype sr_types[ntasks];
  int send_counts[ntasks];
  int send_displs[ntasks];
  int recv_counts[ntasks];
  int recv_displs[ntasks];
  int *recv_buf=NULL;
  int nrecvs;
  int maxreq = DEF_P2P_MAXREQ;
  int ierr;
  int io_comprank;
  int ioindex;
  int tsize;
  int numiotasks;

  
  return ierr;

}
*/

int rearrange_comp2io(const iosystem_desc_t ios, io_desc_t *iodesc, void *sbuf,
			  void *rbuf, const int comm_option, const int fc_options)
{

  bool handshake=true;
  bool isend = false;
  int maxreq = MAX_GATHER_BLOCK_SIZE;
  int ntasks;
  int niotasks;
  int *scount = iodesc->scount;

  int i, tsize;
  int *sendcounts;
  int *recvcounts;
  int *sdispls;
  int *rdispls;
  MPI_Datatype *sendtypes;
  MPI_Datatype *recvtypes;
  MPI_Comm mycomm;
  
  if(iodesc->rearranger == PIO_REARR_BOX){
    mycomm = ios.union_comm;
    niotasks = ios.num_iotasks;
  }else{
    mycomm = iodesc->subset_comm;
    niotasks = 1;
  }  
  MPI_Comm_size(mycomm, &ntasks);

#ifdef _MPISERIAL
  if(iodesc->basetype == 4){
    for(i=0;i<iodesc->llen;i++)
      ((int *) rbuf)[ iodesc->rindex[i] ] = ((int *)sbuf)[ iodesc->sindex[i]];
  }else{
    for(i=0;i<iodesc->llen;i++){
      ((double *) rbuf)[ iodesc->rindex[i] ] = ((double *)sbuf)[ iodesc->sindex[i]];   
    }
    printf("\n");
  }
#else
  define_iodesc_datatypes(ios, iodesc);
  
  sendcounts = (int *) malloc(ntasks*sizeof(int));
  recvcounts = (int *) malloc(ntasks*sizeof(int));
  sdispls = (int *) malloc(ntasks*sizeof(int));
  rdispls = (int *) malloc(ntasks*sizeof(int));
  sendtypes = (MPI_Datatype *) malloc(ntasks*sizeof(MPI_Datatype));
  recvtypes = (MPI_Datatype *) malloc(ntasks*sizeof(MPI_Datatype));

  for(i=0;i<ntasks;i++){
    sendcounts[i] = 0;
    recvcounts[i] = 0; 
    sdispls[i] = 0; 
    rdispls[i] = 0;
    recvtypes[ i ] = MPI_DATATYPE_NULL;
    sendtypes[ i ] =  MPI_DATATYPE_NULL;
  }


  if(ios.ioproc && iodesc->nrecvs>0){
    //    printf("%d: rindex[%d] %d\n",ios.comp_rank,0,iodesc->rindex[0]);
    for( i=0;i<iodesc->nrecvs;i++){
      if(iodesc->rearranger==PIO_REARR_SUBSET){
	recvcounts[ i ] = 1;
	recvtypes[ i ] = iodesc->rtype[i];
      }else{
	recvcounts[ iodesc->rfrom[0] ] = 1;
	recvtypes[ iodesc->rfrom[0] ] = iodesc->rtype[0];
	rdispls[ iodesc->rfrom[0] ] = 0;
	//    printf("%d: rindex[%d] %d\n",ios.comp_rank,0,iodesc->rindex[0]);
	for( i=1;i<iodesc->nrecvs;i++){
	  recvcounts[ iodesc->rfrom[i] ] = 1;
	  recvtypes[ iodesc->rfrom[i] ] = iodesc->rtype[i];

	}
      }
      //   printf("%d: rindex[%d] %d\n",ios.comp_rank,i,iodesc->rindex[i]);

    }
  }

  for( i=0;i<niotasks; i++){
    int io_comprank = ios.ioranks[i];
    if(iodesc->rearranger==PIO_REARR_SUBSET)
      io_comprank=0;
    //    printf("scount[%d]=%d\n",i,scount[i]);
    if(scount[i] > 0) {
      sendcounts[io_comprank]=1;
      sendtypes[io_comprank]=iodesc->stype[i];
    }else{
      sendcounts[io_comprank]=0;
    }
  }      

  // Data in sbuf on the compute nodes is sent to rbuf on the ionodes

  pio_swapm( sbuf,  sendcounts, sdispls, sendtypes,
	     rbuf, recvcounts, rdispls, recvtypes, 
	     mycomm, handshake, isend, maxreq);

  free(sendcounts);
  free(recvcounts); 
  free(sdispls);
  free(rdispls);
  free(sendtypes);
  free(recvtypes);
#endif
  return PIO_NOERR;
}

int rearrange_io2comp(const iosystem_desc_t ios, io_desc_t *iodesc, void *sbuf,
			  void *rbuf, const int comm_option, const int fc_options)
{
  

  bool handshake=true;
  bool isend = false;
  int maxreq = MAX_GATHER_BLOCK_SIZE;
  MPI_Comm mycomm;

  int ntasks ;
  int niotasks;
  int *scount = iodesc->scount;

  int *sendcounts;
  int *recvcounts;
  int *sdispls;
  int *rdispls;
  MPI_Datatype *sendtypes;
  MPI_Datatype *recvtypes;

  int i, tsize;
  if(iodesc->rearranger==PIO_REARR_BOX){
    mycomm = ios.union_comm;
    niotasks = ios.num_iotasks;
  }else{
    mycomm = iodesc->subset_comm;
    niotasks=1;
  }
  MPI_Comm_size(mycomm, &ntasks);

#ifdef _MPISERIAL
  if(iodesc->basetype == 4){
    for(i=0;i<iodesc->llen;i++)
      ((int *) rbuf)[ iodesc->sindex[i] ] = ((int *)sbuf)[ iodesc->rindex[i]];
  }else{
    for(i=0;i<iodesc->llen;i++)
      ((double *) rbuf)[ iodesc->sindex[i] ] = ((double *)sbuf)[ iodesc->rindex[i]];
  }
#else  
  define_iodesc_datatypes(ios, iodesc);

  sendcounts = (int *) calloc(ntasks,sizeof(int));
  recvcounts = (int *) calloc(ntasks,sizeof(int));
  sdispls = (int *) calloc(ntasks,sizeof(int));
  rdispls = (int *) calloc(ntasks,sizeof(int));
  sendtypes = (MPI_Datatype *) malloc(ntasks*sizeof(MPI_Datatype));
  recvtypes = (MPI_Datatype *) malloc(ntasks*sizeof(MPI_Datatype));


  for( i=0;i< ntasks;i++){
    sendtypes[ i ] = MPI_DATATYPE_NULL;
    recvtypes[ i ] = MPI_DATATYPE_NULL;
  }
  if(ios.ioproc){
    for( i=0;i< iodesc->nrecvs;i++){
      if(iodesc->rearranger==PIO_REARR_SUBSET){
	sendcounts[ i ] = 1;
	sendtypes[ i ] = iodesc->rtype[i];
      }else{
	sendcounts[ iodesc->rfrom[i] ] = 1;
	sendtypes[ iodesc->rfrom[i] ] = iodesc->rtype[i];
      }
    }
  }

  for( i=0;i<niotasks; i++){
    int io_comprank = ios.ioranks[i];
    if(iodesc->rearranger==PIO_REARR_SUBSET)
      io_comprank=0;
    if(scount[i] > 0) {
      recvcounts[io_comprank]=1;
      recvtypes[io_comprank]=iodesc->stype[i];
    }
  } 
  
  //
  // Data in sbuf on the ionodes is sent to rbuf on the compute nodes
  //

  pio_swapm( sbuf,  sendcounts, sdispls, sendtypes,
	     rbuf, recvcounts, rdispls, recvtypes, 
	     mycomm, handshake,isend, maxreq);

  free(sendcounts);
  free(recvcounts); 
  free(sdispls);
  free(rdispls);
  free(sendtypes);
  free(recvtypes);
#endif
 
  return PIO_NOERR;

}
// the box rearranger sorts the data from the compute tasks into arrays on the io tasks which will be contiguous on 
// the file.   Each compute tasks may communicate with one or more io tasks 
int box_rearrange_create(const iosystem_desc_t ios,const int maplen, const PIO_Offset compmap[], const int gsize[],
			 const int ndims, io_desc_t *iodesc)
{
  int ierr=PIO_NOERR;
  int ntasks = ios.num_comptasks;
  int numiotasks = ios.num_iotasks;
  PIO_Offset gstride[ndims];
  PIO_Offset iomap;
  PIO_Offset start[ndims], count[ndims];
  int  tsize, i, j, k, llen;
  MPI_Datatype dtype;
  int dest_ioproc[maplen];
  PIO_Offset dest_ioindex[maplen];
  int send_counts[ntasks]; 
  int send_displs[ntasks];
  int recv_counts[ntasks];
  int recv_displs[ntasks];
  MPI_Datatype sr_dtypes[ntasks];
  PIO_Offset iomaplen[numiotasks];
  int maxreq = MAX_GATHER_BLOCK_SIZE;
  PIO_Offset s2rindex[maplen];

  iodesc->rearranger = PIO_REARR_BOX;

  iodesc->ndof = maplen;
  // the stride of each array data dimension in the global (disk) array 
  gstride[ndims-1]=1;
  for(int i=ndims-2;i>=0; i--)
    gstride[i]=gstride[i+1]*gsize[i+1];

#ifdef _MPISERIAL
  tsize = sizeof(PIO_Offset);
#else
  MPI_Type_size(PIO_OFFSET, &tsize);
#endif

  for(i=0; i< maplen; i++){
    dest_ioproc[i] = -1;
    dest_ioindex[i] = -1;
  }

  // These parameters define the communication pattern in pio_swapm
  for(i=0;i<ntasks;i++){
    send_counts[i] = 0;
    send_displs[i] = 0;
    recv_counts[i] = 0;
    recv_displs[i] = 0;
    sr_dtypes[i] = MPI_OFFSET;
  }

  iodesc->llen=0;
  if(ios.ioproc){
    for( i=0;i<ntasks;i++){
      send_counts[ i ] = 1;
    }
    // the start and count arrays for the call to the underlying io library have already been computed in initdecomp
    // The iodescriptor allows for multiple blocks of data (the region linked list) on the io task but the box rerarranger only needs 1. 
    // llen is the total data size on this io task 
    iodesc->llen=1;
    for(i=0;i<ndims;i++)
      iodesc->llen *= iodesc->firstregion->count[i];
  }

  for( i=0;i<numiotasks; i++){
    int io_comprank = ios.ioranks[i];
    recv_counts[ io_comprank ] = 1;
    recv_displs[ io_comprank ] = i*tsize;
  }      

  //  The length of each iomap

  pio_swapm(&(iodesc->llen), send_counts, send_displs, sr_dtypes,
	    iomaplen, recv_counts, recv_displs, sr_dtypes, 	
	    ios.union_comm, false, false, maxreq);

  for(i=0; i<numiotasks; i++){
    if(iomaplen[i]>0){
      int io_comprank = ios.ioranks[i];
      for( j=0; j<ntasks ; j++){
	send_counts[ j ] = 0;
	send_displs[ j ] = 0;
	recv_displs[ j ] = 0;
	recv_counts[ j ] = 0;
	if(ios.union_rank == io_comprank)
	  send_counts[ j ] = ndims;
      }
      recv_counts[ io_comprank ] = ndims;
      
      // The count from iotask i is sent to all compute tasks
      
      pio_swapm(iodesc->firstregion->count,  send_counts, send_displs, sr_dtypes,
		count, recv_counts, recv_displs, sr_dtypes, 
		ios.union_comm, false, false, MAX_GATHER_BLOCK_SIZE);
      
      // The start from iotask i is sent to all compute tasks
      pio_swapm(iodesc->firstregion->start,  send_counts, send_displs, sr_dtypes,
		start, recv_counts, recv_displs, sr_dtypes, 
		ios.union_comm, false, false, MAX_GATHER_BLOCK_SIZE);

      for(k=0;k<maplen;k++){
	PIO_Offset gcoord[ndims], lcoord[ndims];
	bool found=true;
	// For each compmap element find the io task to send to 
	gindex_to_coord(ndims, compmap[k], gstride, gcoord);
	for(j=0;j<ndims;j++){
	  if(gcoord[j] >= start[j] && gcoord[j] < start[j]+count[j]){
	    lcoord[j] = gcoord[j] - start[j];
	  }else{
	    found = false;
	    break;
	  }
	}
	if(found){
	  dest_ioindex[k] = coord_to_lindex(ndims, lcoord, count);
	  dest_ioproc[k] = i;
	}
      }
    }
  }


  for(k=0; k<maplen; k++){
    if(dest_ioproc[k] == -1 && compmap[k]>=0){
      fprintf(stderr,"No destination found for compmap[%d] = %ld\n",k,compmap[k]);
      MPI_Abort(MPI_COMM_WORLD,0);
    }
  }

  
  numiotasks = ios.num_iotasks;

  iodesc->scount = (int *) calloc(numiotasks,sizeof(int));

  // iodesc->scount is the amount of data sent to each io task from the current compute task
  for(i=0;i<maplen; i++){
    if(dest_ioindex[i] != -1){
      (iodesc->scount[dest_ioproc[i]])++;
    }
  }


  for(i=0;i<ntasks;i++){
    send_counts[i] = 0;
    send_displs[i] = 0;
    recv_counts[i] = 0;
    recv_displs[i] = 0;
    sr_dtypes[i] = MPI_INT;
  }

  for(i=0;i<numiotasks;i++){
    int io_comprank;
    io_comprank = ios.ioranks[i];
    send_counts[io_comprank] = 1;
    send_displs[io_comprank] = i*sizeof(int);
  }

  int *recv_buf=NULL;
  if(ios.ioproc){
    recv_buf = (int *) malloc(ntasks * sizeof(int));
    for(i=0;i<ntasks;i++){
      recv_buf[i] = 0;
      recv_counts[i] = 1;
      recv_displs[i] = i*sizeof(int);
    }
  }
  // Share the iodesc->scount from each compute task to all io tasks
  ierr = pio_swapm( iodesc->scount, send_counts, send_displs, sr_dtypes, 
                    recv_buf,  recv_counts, recv_displs, sr_dtypes,
		    ios.union_comm, false, false, maxreq);

  int nrecvs = 0;
  if(ios.ioproc){
    for(i=0;i<ntasks; i++){
      if(recv_buf[i] != 0)
	nrecvs++;
    }

    iodesc->rcount = (int *) calloc(max(1,nrecvs),sizeof(int));
    iodesc->rfrom = (int *) calloc(max(1,nrecvs),sizeof(int));
    

    nrecvs = 0;
    for(i=0;i<ntasks; i++){
      if(recv_buf[i] != 0){
	iodesc->rcount[nrecvs] = recv_buf[i];
	iodesc->rfrom[nrecvs] = i;
	nrecvs++;
      }

    }
    free(recv_buf);
  }

  iodesc->nrecvs = nrecvs;
  if(iodesc->sindex == NULL)
    iodesc->sindex = (PIO_Offset *) calloc(iodesc->ndof,sizeof(PIO_Offset));


  int tempcount[numiotasks];
  int spos[numiotasks];

  spos[0]=0;
  tempcount[0]=0;
  for(i=1;i<numiotasks;i++){
    spos[i] = spos[i-1] + iodesc->scount[i-1];
    tempcount[i]=0;
  }

  for(i=0;i<maplen;i++){
    int iorank = dest_ioproc[i]; 
    if(iorank > -1){
      iodesc->sindex[spos[iorank]+tempcount[iorank]] = i;
      s2rindex[spos[iorank]+tempcount[iorank]] = dest_ioindex[i];

      (tempcount[iorank])++;
    }
  }

  for(i=0;i<ntasks;i++){
    send_counts[i] = 0;
    send_displs[i]  = 0;
    recv_counts[i] = 0;
    recv_displs[i]   =0;
  }
#ifndef _MPISERIAL
  MPI_Type_size(PIO_OFFSET, &tsize);
#else
  tsize = sizeof(PIO_Offset);
#endif
  for(i=0; i<ntasks; i++){
    sr_dtypes[i] = MPI_OFFSET;
  }

  for(i=0;i<numiotasks;i++){
    int io_comprank = ios.ioranks[i];
    send_counts[io_comprank] = iodesc->scount[i];
    if(send_counts[io_comprank]>0)
      send_displs[io_comprank]  = spos[i]*tsize ;
  }

  if(ios.ioproc){
    for(i=0;i<nrecvs;i++)
      recv_counts[iodesc->rfrom[i]] = iodesc->rcount[i];
    recv_displs[0] = 0;
    for(i=1;i<nrecvs;i++)
      recv_displs[iodesc->rfrom[i]] = recv_displs[iodesc->rfrom[i-1]]+iodesc->rcount[i-1]*tsize;
    if(iodesc->llen>0)
      iodesc->rindex = (PIO_Offset *) calloc(iodesc->llen,sizeof(PIO_Offset));
  }


  // s2rindex is the list of indeces on each compute task

  ierr = pio_swapm( s2rindex, send_counts, send_displs, sr_dtypes, 
		    iodesc->rindex, recv_counts, recv_displs, sr_dtypes,
  		    ios.union_comm, true, false, maxreq);


  if(ios.ioproc){
    compute_maxIObuffersize(ios.io_comm, iodesc);
  }

  return ierr;
}

// The sort algorythm used in the subset rearranger
int compare_offsets(const void *a,const void *b) 
{
  mapsort *x = (mapsort *) a;
  mapsort *y = (mapsort *) b;
  return (int) (x->iomap - y->iomap);
}    

//
// Find all regions. 
//
void get_start_and_count_regions(const MPI_Comm io_comm, io_desc_t *iodesc, const int gdims[],const PIO_Offset map[])
{
  int i;
  int nmaplen;
  int regionlen;
  io_region *region;

  nmaplen = 0;
  region = iodesc->firstregion;
  while(map[nmaplen++]<0);
  nmaplen--;
  region->loffset=nmaplen;

  iodesc->maxregions = 1;
  while(nmaplen < iodesc->llen){
    // Here we find the largest region from the current offset into the iomap
    // regionlen is the size of that region and we step to that point in the map array
    // until we reach the end 

    regionlen = find_first_region(iodesc->ndims, gdims, iodesc->llen-nmaplen, 
					map+nmaplen, region->start, region->count);


    for(i=0;i<iodesc->ndims;i++)
      pioassert(region->start[i]>=0,"Failed to find valid region",__FILE__,__LINE__);


    nmaplen = nmaplen+regionlen;
    if(region->next==NULL && nmaplen<iodesc->llen){
      region->next = alloc_region(iodesc->ndims);
      // The offset into the local array buffer is the sum of the sizes of all of the previous regions (loffset) 
      region=region->next;
      region->loffset = nmaplen;
      // The calls to the io library are collective and so we must have the same number of regions on each
      // io task iodesc->maxregions will be the total number of regions on this task 
      iodesc->maxregions++;
    }
  }
  // pad maxregions on all tasks to the maximum and use this to assure that collective io calls are made.
#ifndef _MPISERIAL  
  MPI_Allreduce(MPI_IN_PLACE,&(iodesc->maxregions), 1, MPI_INTEGER, MPI_MAX, io_comm);
#endif
}

void default_subset_partition(const iosystem_desc_t ios, io_desc_t *iodesc)
{
  int taskratio = ios.num_comptasks/ios.num_iotasks;
  int color;
  int key;

  /* Create a new comm for each subset group with the io task in rank 0 and
     only 1 io task per group */

  if(ios.ioproc)
    key=0;
  else
    key=ios.comp_rank%taskratio+1;
  color = ios.comp_rank/taskratio;

  MPI_Comm_split(ios.comp_comm, color, key, &(iodesc->subset_comm));

}

// The subset rearranger sorts the data from a subset of compute tasks onto a 
// single io task.

int subset_rearrange_create(const iosystem_desc_t ios,const int maplen, PIO_Offset compmap[], 
			    const int gsize[], const int ndims, io_desc_t *iodesc)
{

  int taskratio;

  int i, j, jlast;
  bool hs=false;
  bool isend=false;
  PIO_Offset *iomap=NULL;
  int ierr = PIO_NOERR;
  mapsort *map=NULL;

  PIO_Offset *srcindex=NULL;
  PIO_Offset totalfieldsize;

  int maxreq = MAX_GATHER_BLOCK_SIZE;
  int amaplen;
  int rank, ntasks;
  size_t pio_offset_size=sizeof(PIO_Offset);
  /* subset partitions each have exactly 1 io task which is task 0 of that subset_comm */ 
  /* TODO: introduce a mechanism for users to define partitions */
  default_subset_partition(ios, iodesc);

  MPI_Comm_rank(iodesc->subset_comm, &rank);
  MPI_Comm_size(iodesc->subset_comm, &ntasks);

  if(ios.ioproc){
    pioassert(rank==0," io rank in subset comm must be 0",__FILE__,__LINE__);
    iodesc->nrecvs=ntasks;
  }else{
    pioassert(rank>0," bad compute task rank in subset comm",__FILE__,__LINE__);
  }
  int send_counts[ntasks];
  int send_displs[ntasks];
  int recv_counts[ntasks];
  int recv_displs[ntasks];

  MPI_Datatype dtypes[ntasks];

  iodesc->ndof = maplen;
  amaplen=0;
  totalfieldsize=1;
  for(i=0;i<ndims;i++)
    totalfieldsize *= gsize[i];

  // count the data on compute tasks without holes
  for(i=0;i<maplen;i++){
    if(compmap[i]>=0){
      amaplen++;
    }
    pioassert(compmap[i] >= -1 && compmap[i] < totalfieldsize 
	      , " Invalid value in compmap",__FILE__,__LINE__);
  }

  iodesc->scount = (int *) malloc(sizeof(int));
  iodesc->scount[0] = amaplen;

  iodesc->rearranger = PIO_REARR_SUBSET;

  // Create a mapping of indices for communication into the compute task data buffer (without holes)
  // For example if compmap={0,1,2,5,8,-1,14} then maplen=7, amaplen=6 and sindex={0,1,2,3,4,6}

  iodesc->sindex = (PIO_Offset *) calloc(iodesc->scount[0],pio_offset_size); 
  j=0;
  for(i=0;i<maplen;i++){
    if(compmap[i]>=0){
      iodesc->sindex[j++]=i;
    }
  }

  // These arguments control the communication pattern in pio_swapm
  for(i=0;i<ntasks;i++){
    // Number of data items to send to each task
    send_counts[i] = 0;
    // offset in bytes from the beginning of the send array to the first item to send to each task
    send_displs[i] = 0;
    // Number of data items to recieve from each task
    recv_counts[i] = 0;
    // offset in bytes from the beginning of the recieve array to the first item to receve on this task
    recv_displs[i] = 0;
    // Data type (may be different for send and recieve)
    dtypes[i] = MPI_INT;
  }

  send_counts[0] = 1;
  if(ios.ioproc){
    //  rcount is an array on the io tasks of the amount of data to collect from each compute task
    iodesc->rcount = (int *) malloc(ntasks *sizeof(int));
    for(i=0;i<ntasks;i++){
      recv_counts[i]=1;
      if(i>0)
	recv_displs[i] = recv_displs[i-1]+sizeof(int);
    }
  }
  

  // Pass the maplen from each compute task to its associated IO task
  // This is the total count of data elements to be communicated 
  pio_swapm((void *) iodesc->scount, send_counts, send_displs, dtypes, 
	    iodesc->rcount, recv_counts, recv_displs, dtypes, 
	    iodesc->subset_comm, hs, isend, maxreq);

  iodesc->llen = 0;

  if(ios.ioproc){
    // llen is the total count of data on the io task
    for(i=0;i<ntasks;i++){
      iodesc->llen+=iodesc->rcount[i];
    }
    // srcindex collects the sindex arrays from the compute tasks on the io task
    if(iodesc->llen>0){
      srcindex = (PIO_Offset *) calloc(iodesc->llen,pio_offset_size);
    }

    for(i=0;i<ntasks;i++){
      recv_displs[i]=0;
      recv_counts[i]= iodesc->rcount[ i ];
      if(i>0)
	recv_displs[i] = recv_displs[i-1]+ iodesc->rcount[ i-1 ]*pio_offset_size;
    }
  }   
  send_counts[ 0 ] = iodesc->scount[0];
  for(i=0;i<ntasks;i++)
    dtypes[i] = PIO_OFFSET;

  // Pass the sindex from each compute task to its associated IO task
  pio_swapm((void *) iodesc->sindex, send_counts, send_displs, dtypes, 
	    srcindex, recv_counts, recv_displs, dtypes, 
	    iodesc->subset_comm, hs, isend, maxreq);

  // mapsort structure is defined in pio_internal.h
  if(ios.ioproc){
    map = (mapsort *) malloc(iodesc->llen * sizeof(mapsort));    
    iomap = (PIO_Offset *) calloc(iodesc->llen,pio_offset_size);
  }

  // Now pass the compmap from each compute task to the io task, skipping the holes, we copy the array to do this.

  PIO_Offset *shrtmap = (PIO_Offset *) calloc(iodesc->scount[0],pio_offset_size);
  j=0;
  for(i=0;i<maplen;i++)
    if(compmap[i]>=0)
      shrtmap[j++]=compmap[i];
       
  pio_swapm((void *) shrtmap, send_counts, send_displs, dtypes, 
	    iomap, recv_counts, recv_displs, dtypes, 
	    iodesc->subset_comm, hs, isend, maxreq);

  free(shrtmap);

  if(ios.ioproc){
    int pos=0;
    int k=0;
    for(i=0;i<ntasks;i++){
      for(j=0;j<iodesc->rcount[i];j++){
	(map+k)->rfrom = i;
	(map+k)->soffset = srcindex[pos+j];
	(map+k++)->iomap = iomap[pos+j];
      }
      pos += iodesc->rcount[i];
    }

    // sort the mapping, this will transpose the data into IO order        
    qsort(map, iodesc->llen, sizeof(mapsort), compare_offsets); 

    iodesc->rindex = (PIO_Offset *) calloc(iodesc->llen,pio_offset_size);
    iodesc->rfrom = (int *) calloc(iodesc->llen,sizeof(int));

    int cnt[ntasks];
    for(i=0;i<ntasks;i++)
      cnt[i]=0;
    for(i=0;i<iodesc->llen;i++){
      iodesc->rfrom[i] = (map+i)->rfrom;
      iodesc->rindex[i]=i;
      iomap[i]= (map+i)->iomap;
      // Collect the soffset into an array to send back to compute tasks
      srcindex[ recv_displs[iodesc->rfrom[i]] + cnt[iodesc->rfrom[i]]++   ]=(map+i)->soffset;
    }

    pio_swapm((void *) srcindex, recv_counts, recv_displs, dtypes, 
	    iodesc->sindex, send_counts, send_displs, dtypes, 
	    iodesc->subset_comm, hs, isend, maxreq);

    // sindex is now the location on the compute task of each item on the io task 
    // it may not be in asending order (which would imply a transpose)

    get_start_and_count_regions(ios.io_comm,iodesc,gsize,iomap);

    if(iomap != NULL)
      free(iomap);

    if(map != NULL)
      free(map);

    compute_maxIObuffersize(ios.io_comm, iodesc);
  }

  /*
  MPI_Barrier(ios.union_comm);
  MPI_Finalize();
  exit(-2);
  */
  return ierr;


}
  
    
  
