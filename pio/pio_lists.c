#include <pio.h>
#include <pio_internal.h>
#include <string.h>
#include <stdio.h>
#include <malloc.h>

static iosystem_desc_t *pio_iosystem_list = NULL;
static file_desc_t *pio_file_list = NULL;
static file_desc_t *current_file=NULL;

void pio_add_to_file_list(file_desc_t *file)
{
  file_desc_t *cfile;
  //  assert(file != NULL);

  file->next = NULL;
  if(pio_file_list == NULL)
    pio_file_list = file;
  else{
    for(cfile = pio_file_list; cfile->next != NULL; cfile=cfile->next);
    cfile->next = file;
  }
  current_file = file;
}

     
file_desc_t *pio_get_file_from_id(int ncid)
{
  file_desc_t *cfile;

  cfile = NULL;

  if(current_file != NULL && current_file->fh == ncid)
    cfile=current_file;
  for(cfile=pio_file_list; cfile != NULL; cfile=cfile->next){
    if(cfile->fh == ncid){
      current_file = cfile;
      break;
    }
  }
  return cfile;
}
  
int pio_delete_file_from_list(int ncid)
{

  file_desc_t *cfile, *pfile;

  pfile = NULL;
  for(cfile=pio_file_list; cfile != NULL; cfile=cfile->next){
    if(cfile->fh == ncid){
      if(pfile == NULL){
	pio_file_list = cfile->next;
      }else{
	pfile->next = cfile->next;
      }
      free(cfile);
      return PIO_NOERR;
    }
    pfile = cfile;
  }
  return PIO_EBADID;
}

int pio_delete_iosystem_from_list(int piosysid)
{

  iosystem_desc_t *ciosystem, *piosystem;

  piosystem = NULL;
  for(ciosystem=pio_iosystem_list; ciosystem != NULL; ciosystem=ciosystem->next){
    if(ciosystem->iosysid == piosysid){
      if(piosystem == NULL){
	pio_iosystem_list = ciosystem->next;
      }else{
	piosystem->next = ciosystem->next;
      }
      free(ciosystem);
      return PIO_NOERR;
    }
    piosystem = ciosystem;
  }
  return PIO_EBADID;
}

int pio_add_to_iosystem_list(iosystem_desc_t *ios)
{
  iosystem_desc_t *cios;
  int i=1;
  //  assert(ios != NULL);

  ios->next = NULL;
  if(pio_iosystem_list == NULL){
    pio_iosystem_list = ios;
  }else{
    for(cios = pio_iosystem_list; cios->next != NULL; cios=cios->next, i++);
    cios->next = ios;
  }
  ios->iosysid = i << 16;
  return ios->iosysid;
}

iosystem_desc_t *pio_get_iosystem_from_id(int iosysid)
{
  iosystem_desc_t *ciosystem;

  for(ciosystem=pio_iosystem_list; ciosystem != NULL; ciosystem=ciosystem->next){
    if(ciosystem->iosysid == iosysid){
      return ciosystem;
    }
  }
  return NULL;
  
}
