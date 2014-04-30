#include <pio.h>
#include <pio_internal.h>
#define DEF_P2P_MAXREQ 64

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
  int iosize, totiosize;
  int i;
  io_region *region;

  //  compute the max io buffer size, for conveneance it is the combined size of all regions
  totiosize=0;
  region = iodesc->firstregion;
  while(region != NULL){
    if(region->count[0]>0)
      iosize=1;
      for(i=0;i<iodesc->ndims;i++)
	iosize*=region->count[i];
      totiosize+=iosize;
    region = region->next;
  }
  iodesc->llen = totiosize;
  // Share the max io buffer size with all io tasks
#ifndef _MPISERIAL
  CheckMPIReturn(MPI_Allreduce(MPI_IN_PLACE, &totiosize, 1, MPI_INT, MPI_MAX, io_comm),__FILE__,__LINE__);
#endif
  
  iodesc->maxiobuflen = totiosize;
  //  printf("%d iosize %d %d\n",__LINE__,totiosize, iodesc->llen);
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
      if (map[j + i*region_size] != map[j] + i*stride) {
        can_expand = 0;
        break;
      }
    }
    if (!can_expand) break;
  }
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




int create_mpi_datatypes(const MPI_Datatype basetype,const int msgcnt,const PIO_Offset dlen, const PIO_Offset mindex[],const int mcount[],MPI_Datatype mtype[])
{
  PIO_Offset bsizeT[msgcnt];
  int pos;
  int ii;
  PIO_Offset i8blocksize;
  MPI_Datatype newtype;
  int blocksize;
  PIO_Offset lindex[dlen];
  
  memcpy(lindex, mindex, (size_t) (dlen*sizeof(PIO_Offset)));

  bsizeT[0]=0;
  mtype[0] = MPI_DATATYPE_NULL;
  pos = 0;
  ii = 0;
  if(msgcnt>0){
    for(int i=0;i<msgcnt;i++){
      //      printf("mcount %d lindex %ld\n",mcount[i],lindex[i]);
      if(mcount[i]>0){
	bsizeT[ii] = GCDblocksize(mcount[i], lindex+pos);
	//		for(int j=0;j<mcount[i];j++)
	//		  printf(" %d ",lindex[pos+j]);
	//		printf("\n bsizet[%d] %ld\n",ii,bsizeT[ii]);
	ii++;
	pos+=mcount[i];
      }
    }
    blocksize = (int) lgcd_array(ii ,bsizeT);

    //    printf("blocksize = %d %d\n",blocksize, msgcnt);
    
#ifndef _MPISERIAL
    if(blocksize>1){
      CheckMPIReturn(MPI_Type_contiguous(blocksize, basetype, &newtype),__FILE__,__LINE__);
    }else{
      CheckMPIReturn(MPI_Type_dup(basetype, &newtype), __FILE__,__LINE__);
    }
    CheckMPIReturn(MPI_Type_commit(&newtype), __FILE__,__LINE__);
#endif     

    pos = 0;
    for(int i=0;i< msgcnt; i++){
      //      printf("lindex[%d] %d mcount[%d] %d\n",i,lindex[i],i,mcount[i]);
      if(mcount[i]>0){
	int len = mcount[i]/blocksize;
	int displace[len];
	if(blocksize==1)
	  for(int j=0;j<mcount[i];j++)
	    displace[j] = (int) (lindex+pos)[j];
	else{
	  for(int j=0;j<mcount[i];j++)
	    (lindex+pos)[j]++;
	  for(int j=0;j<len;j++){
	    displace[j]= ((lindex+pos)[j*blocksize]-1)/blocksize;
	    //	    printf("displace[%d] %d pos %d lindex %d blocksize %d\n", j, displace[j],pos, (lindex+pos)[j*blocksize],blocksize);
	  }
	}
#ifndef _MPISERIAL
	CheckMPIReturn(MPI_Type_create_indexed_block(len, 1, displace, newtype, mtype+i),__FILE__,__LINE__);
	CheckMPIReturn(MPI_Type_commit(mtype+i), __FILE__,__LINE__);
	pos+=mcount[i];
#endif
      }
    }
#ifndef _MPISERIAL
    CheckMPIReturn(MPI_Type_free(&newtype),__FILE__,__LINE__);
#endif
  }
  
  return PIO_NOERR;

}


int define_iodesc_datatypes(const iosystem_desc_t ios, io_desc_t *iodesc)
{
  if(ios.ioproc){
    //    printf("%d IO:\n",ios.io_rank);
    if(iodesc->rtype==NULL){
      iodesc->rtype = (MPI_Datatype *) malloc(max(1,iodesc->nrecvs)*sizeof(MPI_Datatype));
      /*      
      printf("rindex: \n");
      for(int i=0;i<iodesc->llen;i++)
	printf("%d ",iodesc->rindex[i]);
      printf("\n");
      for(int i=0;i<iodesc->nrecvs;i++)
	printf("%d rcount %d \n",i,iodesc->rcount[i]);
      */      

     create_mpi_datatypes(iodesc->basetype, iodesc->nrecvs, iodesc->llen, iodesc->rindex, iodesc->rcount, iodesc->rtype);
    }
  }

  
  //printf("COMP:\n");

  if(iodesc->stype==NULL){
    iodesc->stype = (MPI_Datatype *) malloc(ios.num_iotasks*sizeof(MPI_Datatype));
    create_mpi_datatypes(iodesc->basetype, ios.num_iotasks, iodesc->ndof, iodesc->sindex, iodesc->scount, iodesc->stype);
  }

  return PIO_NOERR;

}




int compute_counts(const iosystem_desc_t ios, io_desc_t *iodesc, const int dest_ioproc[], 
		   const PIO_Offset dest_ioindex[])
{
  int niotasks = ios.num_iotasks;
  int ncomptasks = ios.num_comptasks;
  int i;
  int iorank;
  MPI_Datatype sr_types[ncomptasks];
  int send_counts[ncomptasks];
  int send_displs[ncomptasks];
  int recv_counts[ncomptasks];
  int recv_displs[ncomptasks];
  int *recv_buf=NULL;
  int *rcount;
  int *rfrom;
  int nrecvs;
  int pio_maxreq;  
  int ierr;
  int io_comprank;
  int ioindex;
  int tsize;
  int ndof= iodesc->ndof;


  pio_maxreq = DEF_P2P_MAXREQ;

  iodesc->scount = (int *) calloc(niotasks,sizeof(int));

  // iodesc->scount is the amount of data sent to each task from the current task
  for(i=0;i<ndof; i++){
    iorank = dest_ioproc[i];
    if(iorank != -1){
      (iodesc->scount[iorank])++;
    }
  }

  //  for(i=0;i<niotasks;i++)
  //   printf("iodesc->scount = %d\n",iodesc->scount[i]);

  for(i=0;i<ncomptasks;i++){
    send_counts[i] = 0;
    send_displs[i] = 0;
    recv_counts[i] = 0;
    recv_displs[i] = 0;
    sr_types[i] = MPI_INT;
  }

  for(i=0;i<niotasks;i++){
    int io_comprank = ios.ioranks[i];
    send_counts[io_comprank] = 1;
    send_displs[io_comprank] = i*sizeof(int);
  }


  if(ios.ioproc){
    recv_buf = (int *) malloc(ncomptasks * sizeof(int));
    for(i=0;i<ncomptasks;i++){
      recv_buf[i] = 0;
      recv_counts[i] = 1;
      recv_displs[i] = i*sizeof(int);
    }
  }

  // Share the iodesc->scount from each compute task to all io tasks
  ierr = pio_swapm( iodesc->scount, send_counts, send_displs, sr_types, 
                    recv_buf,  recv_counts, recv_displs, sr_types,
		    ios.union_comm, false, false, pio_maxreq);

  nrecvs = 0;
  if(ios.ioproc){
    //    printf("recv_buf = ");
    for(i=0;i<ncomptasks; i++){
      //   printf(" %d ",recv_buf[i]);
      if(recv_buf[i] != 0)
	nrecvs++;
    }
    // printf("\n");

    iodesc->rcount = (int *) calloc(max(1,nrecvs),sizeof(int));
    rcount = iodesc->rcount;
    iodesc->rfrom = (int *) calloc(max(1,nrecvs),sizeof(int));
    
    rfrom = iodesc->rfrom;
    nrecvs = 0;
    for(i=0;i<ncomptasks; i++){
      if(recv_buf[i] != 0){
	rcount[nrecvs] = recv_buf[i];
	rfrom[nrecvs] = i;
	nrecvs++;
      }

    }
  }else{
    rcount = (int *) malloc(sizeof(int));
    rcount[0]=0;
  }

  iodesc->nrecvs = nrecvs;
  iodesc->sindex = (PIO_Offset *) calloc(ndof,sizeof(PIO_Offset));
  PIO_Offset s2rindex[ndof];

  int tempcount[niotasks];
  int spos[niotasks];

  spos[0]=0;
  tempcount[0]=0;
  for(i=1;i<niotasks;i++){
    spos[i] = spos[i-1] + iodesc->scount[i-1];
    tempcount[i]=0;
  }
  for(i=0;i<ndof;i++){
    iorank =dest_ioproc[i]; 
    ioindex = dest_ioindex[i];
    //    printf("%d iorank %d ioindex %ld\n",ios.comp_rank,iorank,ioindex);
    if(iorank > -1){
      iodesc->sindex[spos[iorank]+tempcount[iorank]] = i;
      s2rindex[spos[iorank]+tempcount[iorank]] = ioindex;
      (tempcount[iorank])++;
    }
  }

  for(i=0;i<ncomptasks;i++){
    send_counts[i] = 0;
    send_displs[i]  = 0;
    recv_counts[i] = 0;
    recv_displs[i]   =0;
  }
#ifndef _MPISERIAL
  MPI_Type_size(MPI_OFFSET, &tsize);
#else
  tsize = sizeof(long long);
#endif
  for(i=0; i<ncomptasks; i++){
    sr_types[i] = MPI_OFFSET;
  }
  for(i=0;i<niotasks;i++){
    io_comprank = ios.ioranks[i];
    send_counts[io_comprank] = iodesc->scount[i];
    if(send_counts[io_comprank]>0)
      send_displs[io_comprank]  = spos[i]*tsize ;
  }

  if(ios.ioproc){
    for(i=0;i<nrecvs;i++)
      recv_counts[rfrom[i]] = rcount[i];
    recv_displs[0] = 0;
    for(i=1;i<nrecvs;i++)
      recv_displs[rfrom[i]] = recv_displs[rfrom[i-1]]+rcount[i-1]*tsize;
    if(iodesc->llen>0)
      iodesc->rindex = (PIO_Offset *) calloc(iodesc->llen,sizeof(PIO_Offset));
  }

  //  printf("%d rbuf_size %d\n",ios.comp_rank,rbuf_size);


  // s2rindex is the list of indeces on each compute task
  /*    
  printf("%d s2rindex: ", ios.comp_rank);
  for(i=0;i<ndof;i++)
    printf("%ld ",s2rindex[i]);
  printf("\n");
  */

  ierr = pio_swapm( s2rindex, send_counts, send_displs, sr_types, 
		    iodesc->rindex, recv_counts, recv_displs, sr_types,
		    ios.union_comm, true, false, MAX_GATHER_BLOCK_SIZE);

  //  rindex is an array of the indices of the data to be sent from
  //  this io task to each compute task. 
  /* 
  if(ios.ioproc){
    printf("%d rindex: ",ios.io_rank);
    for(int j=0;j<iodesc->llen;j++)
      printf(" %ld ",iodesc->rindex[j]);
    printf("\n");

    for(int j=0;j<nrecvs;j++){
      printf("%d rfrom %d ",ios.io_rank,rfrom[j]);
      if(j==0)
	for(i=0;i<rcount[j];i++)
	  printf("%ld ",iodesc->rindex[i]);
      else  
	for(i=0;i<rcount[j];i++)
	  printf("%ld ",iodesc->rindex[rcount[j-1]+i]);
      printf("\n");
    }
  }
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Abort(MPI_COMM_WORLD,0);
*/  
  iodesc->rtype = NULL;
  iodesc->stype = NULL;

  return ierr;

}

int box_rearrange_comp2io(const iosystem_desc_t ios, io_desc_t *iodesc, void *sbuf,
			  void *rbuf, const int comm_option, const int fc_options)
{

  bool handshake=true;
  bool isend = false;
  int maxreq = MAX_GATHER_BLOCK_SIZE;
  int nprocs = ios.num_comptasks;
  int *scount = iodesc->scount;

  int i, tsize;
  int *sendcounts;
  int *recvcounts;
  int *sdispls;
  int *rdispls;
  MPI_Datatype *sendtypes;
  MPI_Datatype *recvtypes;

  define_iodesc_datatypes(ios, iodesc);

  sendcounts = (int *) malloc(nprocs*sizeof(int));
  recvcounts = (int *) malloc(nprocs*sizeof(int));
  sdispls = (int *) malloc(nprocs*sizeof(int));
  rdispls = (int *) malloc(nprocs*sizeof(int));
  sendtypes = (MPI_Datatype *) malloc(nprocs*sizeof(MPI_Datatype));
  recvtypes = (MPI_Datatype *) malloc(nprocs*sizeof(MPI_Datatype));

  for(i=0;i<nprocs;i++){
    sendcounts[i] = 0;
    recvcounts[i] = 0; 
    sdispls[i] = 0; 
    rdispls[i] = 0;
    recvtypes[ i ] = MPI_DATATYPE_NULL;
    sendtypes[ i ] =  MPI_DATATYPE_NULL;
  }


  if(ios.ioproc && iodesc->nrecvs>0){
    recvcounts[ iodesc->rfrom[0] ] = 1;
    recvtypes[ iodesc->rfrom[0] ] = iodesc->rtype[0];
    rdispls[ iodesc->rfrom[0] ] = 0;
    //    printf("%d: rindex[%d] %d\n",ios.comp_rank,0,iodesc->rindex[0]);
    for( i=1;i<iodesc->nrecvs;i++){
      recvcounts[ iodesc->rfrom[i] ] = 1;
      recvtypes[ iodesc->rfrom[i] ] = iodesc->rtype[i];

      //   printf("%d: rindex[%d] %d\n",ios.comp_rank,i,iodesc->rindex[i]);

    }
  }else{
    for( i=0;i<iodesc->nrecvs;i++){
      recvcounts[ iodesc->rfrom[i] ] = 0;
    }
  }  

  for( i=0;i<ios.num_iotasks; i++){
    int io_comprank = ios.ioranks[i];
    //    printf("scount[%d]=%d\n",i,scount[i]);
    if(scount[i] > 0) {
      sendcounts[io_comprank]=1;
      sendtypes[io_comprank]=iodesc->stype[i];
    }else{
      sendcounts[io_comprank]=0;
    }
  }      

  // Data in sbuf on the compute nodes is sent to rbuf on the ionodes
  //  printf("%d sbuf %d %d %d\n",ios.comp_rank,((int *)sbuf)[0],((int *)sbuf)[1],((int *)sbuf)[2]);

  pio_swapm( sbuf,  sendcounts, sdispls, sendtypes,
	     rbuf, recvcounts, rdispls, recvtypes, 
	     ios.union_comm, handshake, isend, maxreq);

  // if(rbuf!=NULL)
  //   printf("%d rbuf %d %d %d\n",ios.io_rank,((int *)rbuf)[0],((int *)rbuf)[1],((int *)rbuf)[2]);


  free(sendcounts);
  free(recvcounts); 
  free(sdispls);
  free(rdispls);
  free(sendtypes);
  free(recvtypes);
  return PIO_NOERR;
}

int box_rearrange_io2comp(const iosystem_desc_t ios, io_desc_t *iodesc, void *sbuf,
			  void *rbuf, const int comm_option, const int fc_options)
{
  

  bool handshake=true;
  bool isend = false;
  int maxreq = MAX_GATHER_BLOCK_SIZE;
  int nprocs = ios.num_comptasks;
  int *scount = iodesc->scount;

  int *sendcounts;
  int *recvcounts;
  int *sdispls;
  int *rdispls;
  MPI_Datatype *sendtypes;
  MPI_Datatype *recvtypes;

  int i, tsize;
  
  define_iodesc_datatypes(ios, iodesc);

  sendcounts = (int *) calloc(nprocs,sizeof(int));
  recvcounts = (int *) calloc(nprocs,sizeof(int));
  sdispls = (int *) calloc(nprocs,sizeof(int));
  rdispls = (int *) calloc(nprocs,sizeof(int));
  sendtypes = (MPI_Datatype *) malloc(nprocs*sizeof(MPI_Datatype));
  recvtypes = (MPI_Datatype *) malloc(nprocs*sizeof(MPI_Datatype));


  for( i=0;i< nprocs;i++){
    sendtypes[ i ] = MPI_DATATYPE_NULL;
    recvtypes[ i ] = MPI_DATATYPE_NULL;
  }
  if(ios.ioproc){
    for( i=0;i< iodesc->nrecvs;i++){
      sendcounts[ iodesc->rfrom[i] ] = 1;
      sendtypes[ iodesc->rfrom[i] ] = iodesc->rtype[i];
    }
  }
    

  for( i=0;i<ios.num_iotasks; i++){
    int io_comprank = ios.ioranks[i];
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
	     ios.union_comm, handshake,isend, maxreq);

  free(sendcounts);
  free(recvcounts); 
  free(sdispls);
  free(rdispls);
  free(sendtypes);
  free(recvtypes);

  return PIO_NOERR;

}
/*
void PIO_Offset_size(MPI_Datatype *dtype, int *tsize)
{
  int  tsizei, tsizel;
#ifndef _MPISERIAL   
  MPI_Type_dup(MPI_OFFSET, dtype);
  MPI_Type_size(*dtype,tsize);
  return;
  MPI_Type_size(MPI_LONG_LONG, &tsizel);
  MPI_Type_size(MPI_INT, &tsizei);

  if(sizeof(PIO_Offset) == tsizei){
    *dtype = MPI_INT;
    *tsize = tsizei;
  }else if(sizeof(PIO_Offset) == tsizel){
    *dtype = MPI_LONG_LONG;
    *tsize = tsizel;
  }

}
*/


int box_rearrange_create(const iosystem_desc_t ios,const int maplen, const PIO_Offset compmap[], const int gsize[],
			 const int ndims, io_desc_t *iodesc)
{
  int ierr=PIO_NOERR;
  int nprocs = ios.num_comptasks;
  int nioprocs = ios.num_iotasks;
  PIO_Offset gstride[ndims];
  PIO_Offset iomap;
  PIO_Offset start[ndims], count[ndims];
  int  tsize, i, j, k, llen;
  MPI_Datatype dtype;
  int *dest_ioproc;
  PIO_Offset *dest_ioindex;
  int *sndlths; 
  int *sdispls;
  int *recvlths;
  int *rdispls;
  MPI_Datatype *dtypes;

  dest_ioproc = (int *) malloc(maplen*sizeof(int));
  dest_ioindex = (PIO_Offset *) malloc(maplen*sizeof(PIO_Offset));

  iodesc->rearranger = PIO_REARR_BOX;

  iodesc->ndof = maplen;
  gstride[ndims-1]=1;
  for(int i=ndims-2;i>=0; i--)
    gstride[i]=gstride[i+1]*gsize[i+1];

  //  PIO_Offset_size(&dtype, &tsize);
#ifndef _MPISERIAL
  MPI_Type_size(MPI_OFFSET, &tsize);
#endif
  sndlths = (int *) malloc(nprocs*sizeof(int)); 
  sdispls= (int *) malloc(nprocs*sizeof(int));
  recvlths= (int *) malloc(nprocs*sizeof(int));
  rdispls= (int *) malloc(nprocs*sizeof(int));
  dtypes= (MPI_Datatype *) malloc(nprocs*sizeof(MPI_Datatype));


  for(i=0; i< maplen; i++){
    dest_ioproc[i] = -1;
    dest_ioindex[i] = 0;
  }
  for(i=0;i<nprocs;i++){
    sndlths[i] = 0;
    sdispls[i] = 0;
    recvlths[i] = 0;
    rdispls[i] = 0;
    dtypes[i] = MPI_OFFSET;
  }
  if(ios.ioproc){
    for( i=0;i<nprocs;i++){
      sndlths[ i ] = 1;
    }
  }
  for( i=0;i<nioprocs; i++){
    int io_comprank = ios.ioranks[i];
    recvlths[ io_comprank ] = 1;
    rdispls[ io_comprank ] = i*tsize;
  }      
  PIO_Offset iomaplen[nioprocs];
  //  The length of each iomap
  pio_swapm(&(iodesc->llen), sndlths, sdispls, dtypes,
	    iomaplen, recvlths, rdispls, dtypes, 	
	    ios.union_comm, false, false, MAX_GATHER_BLOCK_SIZE);


  for(i=0; i<nioprocs; i++){
    if(iomaplen[i]>0){
      int io_comprank = ios.ioranks[i];
      for( j=0; j<nprocs ; j++){
	sndlths[ j ] = 0;
	sdispls[ j ] = 0;
	rdispls[ j ] = 0;
	recvlths[ j ] = 0;
	if(ios.union_rank == io_comprank)
	  sndlths[ j ] = ndims;
      }
      recvlths[ io_comprank ] = ndims;
      
      // The count from iotask i is sent to all compute tasks
      


      pio_swapm(iodesc->firstregion->count,  sndlths, sdispls, dtypes,
		count, recvlths, rdispls, dtypes, 
		ios.union_comm, false, false, MAX_GATHER_BLOCK_SIZE);
      
      // The start from iotask i is sent to all compute tasks
      pio_swapm(iodesc->firstregion->start,  sndlths, sdispls, dtypes,
		start, recvlths, rdispls, dtypes, 
		ios.union_comm, false, false, MAX_GATHER_BLOCK_SIZE);

      for(k=0;k<maplen;k++){
	PIO_Offset gcoord[ndims], lcoord[ndims];
	bool found=true;
	gindex_to_coord(ndims, compmap[k], gstride, gcoord);
	for(j=0;j<ndims;j++){
	  //	  printf("%d %d map %d gcoord %d start %d count %d\n",j,k,compmap[k],gcoord[j],start[j],count[j]);
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
  //    printf("dest_ioproc %d %d %d dest_ioindex %d %d %d\n",dest_ioproc[0],dest_ioproc[1],dest_ioproc[2],
  // 	 dest_ioindex[0],dest_ioindex[1],dest_ioindex[2]);

  for(k=0; k<maplen; k++){
    if(dest_ioproc[k] == -1 && compmap[k]>=0){
      fprintf(stderr,"No destination found for compmap[%d] = %ld\n",k,compmap[k]);
      MPI_Abort(MPI_COMM_WORLD,0);
    }
  }

  compute_counts(ios, iodesc, dest_ioproc, dest_ioindex);

  return PIO_NOERR;
}

typedef struct mapsort
{
  int rfrom;
  int soffset;
  PIO_Offset iomap;
} mapsort;

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
  region->loffset=0;

  iodesc->maxregions = 1;
  while(nmaplen < iodesc->llen){
    // Here we find the largest region from the current offset into the iomap
    // regionlen is the size of that region and we step to that point in the map array
    // until we reach the end 
    regionlen = find_first_region(iodesc->ndims, gdims, iodesc->llen-nmaplen, 
					map+nmaplen, region->start, region->count);

    //    printf("start %ld %ld %ld\n", region->start[0], region->start[1], region->start[2]);
    // printf("count %ld %ld %ld\n", region->count[0], region->count[1], region->count[2]);

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





int subset_rearrange_create(const iosystem_desc_t ios,const int maplen, const PIO_Offset compmap[], 
			    const int gsize[], const int ndims, io_desc_t *iodesc)
{
  int *dest_ioproc=NULL;
  PIO_Offset *dest_ioindex=NULL;
  int tsize;
  MPI_Datatype dtype;
  int taskratio;
  int nprocs = ios.num_comptasks;
  int nioprocs = ios.num_iotasks;
  int *sndlths; 
  int *sdispls;
  int *recvlths;
  int *rdispls;
  MPI_Datatype *dtypes;
  int i, j, jlast;
  bool hs=false;
  bool isend=false;
  PIO_Offset *iomap=NULL;
  int ierr = PIO_NOERR;
  mapsort *map=NULL;
  int ctoiotask;
  int destloc;
  PIO_Offset *destoffset=NULL;
  PIO_Offset gstride[ndims];
  int dioproc;

  if(maplen>0){
    dest_ioproc = (int *) malloc(maplen*sizeof(int));
    dest_ioindex = (PIO_Offset *) malloc(maplen*sizeof(PIO_Offset));
  }
  iodesc->ndof = maplen;
  iodesc->rearranger = PIO_REARR_SUBSET;

  //  PIO_Offset_size(&dtype, &tsize);
#ifndef _MPISERIAL
  MPI_Type_size(MPI_INT, &tsize);
#else
  tsize = sizeof(int);
#endif
  taskratio = nprocs/nioprocs;

  gstride[ndims-1]=1;
  for(i=ndims-2;i>=0; i--)
    gstride[i]=gstride[i+1]*gsize[i+1];

  // Each compute task sends to only one IO task
  dioproc = ios.comp_rank/taskratio;
  for(i=0;i<maplen;i++){
    dest_ioproc[i] = dioproc;
  }
  // Pass the maplen from each compute task to its associated IO task
  
  sndlths = (int *) malloc(nprocs*sizeof(int)); 
  sdispls= (int *) malloc(nprocs*sizeof(int));
  recvlths= (int *) malloc(nprocs*sizeof(int));
  rdispls= (int *) malloc(nprocs*sizeof(int));
  dtypes= (MPI_Datatype *) malloc(nprocs*sizeof(MPI_Datatype));

  for(i=0;i<nprocs;i++){
    sndlths[i] = 0;
    sdispls[i] = 0;
    recvlths[i] = 0;
    rdispls[i] = 0;
    dtypes[i] = MPI_INT;
  }


  
  sndlths[ ios.ioranks[ dioproc ] ] = 1;
  if(ios.ioproc){
    for(i=0;i<nprocs;i++)
      if(ios.io_rank == i/taskratio){
	recvlths[i]=1;
	rdispls[i] = sizeof(int) * (i % taskratio);
      }
  }
  int recvlen[taskratio];
  for(i=0;i<taskratio;i++)
    recvlen[i]=0;

  pio_swapm(&maplen, sndlths, sdispls, dtypes, 
	    recvlen, recvlths, rdispls, dtypes, 
	    ios.union_comm, hs, isend, MAX_GATHER_BLOCK_SIZE);

  // Now pass the map
  iodesc->llen = 0;
  if(ios.ioproc){
    for(i=0;i<taskratio;i++){
      iodesc->llen+=recvlen[i];
    }
    if(iodesc->llen>0){
      map = (mapsort *) malloc(iodesc->llen * sizeof(mapsort));    
      iomap = (PIO_Offset *) calloc(iodesc->llen,sizeof(PIO_Offset));
      destoffset = (PIO_Offset *) calloc(iodesc->llen,sizeof(PIO_Offset));
    }
  }
  sndlths[ ios.ioranks[ dioproc ] ] = maplen;
  if(ios.ioproc){
    jlast =0;
    for(i=0;i<nprocs;i++){
      ctoiotask = i%taskratio;
      rdispls[i]=0;
      if(ios.io_rank == i/taskratio){
	recvlths[i]= recvlen[ ctoiotask ];

	if(i>0)
	  rdispls[i] = rdispls[i-1]+recvlths[i-1]*sizeof(PIO_Offset);
	for(j=jlast;j<jlast+recvlths[i];j++){
	  (map+j)->rfrom = i;
	  (map+j)->soffset = j-jlast;
	}
	jlast = j;
      }
    }
      
  }
  
  for(i=0;i<nprocs;i++){
    dtypes[i] = dtype;
    //    printf("rdispls[%d] %d\n",i,rdispls[i]);
  }
  //    for(i=0;i<maplen;i++)
  //    printf("compmap[%d] %ld\n",i,compmap[i]);

  pio_swapm(compmap, sndlths, sdispls, dtypes, 
	    iomap, recvlths, rdispls, dtypes, 
	    ios.union_comm, hs, true, MAX_GATHER_BLOCK_SIZE);


  if(ios.ioproc){
    int cnt[nprocs];
    int sender;

    for(i=0;i<nprocs;i++)
      cnt[i]=0;

    for(i=0;i<iodesc->llen;i++){
      (map+i)->iomap = iomap[i];
      //     printf("iomap[%d] %ld ",i,iomap[i]);
    }
    //printf("\n");

     // sort the mapping, this will transpose the data into IO order        
    qsort(map, iodesc->llen, sizeof(mapsort), compare_offsets); 

    for(i=0;i<iodesc->llen;i++){
      iomap[i]=(map+i)->iomap;
      sender = (map+i)->rfrom;
      destoffset[ rdispls[sender]/sizeof(PIO_Offset) + cnt[sender]++ ] = i;
      //      printf("%d %d displs %d destoffset[%d] %ld\n",ios.io_rank,sender,rdispls[sender]/sizeof(PIO_Offset),i,destoffset[i]);
    }

    get_start_and_count_regions(ios.io_comm,iodesc,gsize,iomap);

    compute_maxIObuffersize(ios.io_comm, iodesc);

    free(map);
    free(iomap);


  }
  //  printf("%d %d\n",__LINE__,recvlths[1]);
  pio_swapm(destoffset, recvlths, rdispls, dtypes, 
	    dest_ioindex, sndlths, sdispls, dtypes, 
	    ios.union_comm, hs, isend, MAX_GATHER_BLOCK_SIZE);

  compute_counts(ios, iodesc, dest_ioproc, dest_ioindex);

  free(sndlths); 
  free(sdispls);
  free(recvlths);
  free(rdispls);
  free(dtypes);

  free(dest_ioproc);
  free(dest_ioindex);

  if(ios.ioproc)
    free(destoffset);



  return ierr;


}
  
    
  
