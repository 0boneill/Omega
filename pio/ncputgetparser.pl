#!/usr/bin/perl 
use strict;

my $netcdf_inc = "$ENV{NETCDF}/include/netcdf.h";
my $pnetcdf_inc = "$ENV{PNETCDF}/include/pnetcdf.h";

my $functions;

open(PF,$pnetcdf_inc) or die "Could not open $pnetcdf_inc";
my @file = <PF>;
my $func="";
foreach my $line (@file){
  chomp($line);
  next if($line =~ /^\s*\/\*/);
  next if($line =~ /^\s*[#\*]/);
  $line =~ s/\s+/ /g; 
  if($line =~ / ncmpi_(.*)(\(.*)$/) {
    $func = $1;

    $functions->{$func}{pnetcdf} = $2;
  }elsif($func ne ""){
    $functions->{$func}{pnetcdf} .= $line;
  }
  $func="" if($line =~ /\;\s*$/);
}
close(PF);

open(NF,$netcdf_inc) or die "Could not open $netcdf_inc";
my @file = <NF>;

my $func="";
foreach my $line (@file){
  chomp($line);
  next if($line =~ /^\s*\/\*/);
  next if($line =~ /^\s*[#\*]/); 
  $line =~ s/\s+/ /g;
  if($line =~ /^\s*nc_([^_].*)(\(.*)$/) {
    $func = $1;
    $functions->{$func}{netcdf} = $2;
  }elsif($func ne ""){
    $functions->{$func}{netcdf} .= $line;
  }
  $func="" if($line =~ /\;\s*$/);
}
close(NF);

my $func="";

open(F,">pio_putget_nc.c");
print F "#include <pio.h>
#include <pio_internal.h>\n\n";
#print "$functions->{inq}{pnetcdf}\n";
#print "$functions->{inq}{netcdf}\n";
#print "$functions->{inquire}{pio}\n";
  open(T,"pio_c_putget_template.c") or die "Could not open file pio_c_putget_template.c";
  my @template = <T>;
  close(T);

foreach my $func (keys %{$functions}){
  next unless($func=~/get_v/ or $func=~/put_v/);

  my @tmptmp = @template;

  if($functions->{$func}{pnetcdf} =~ /ngattsp/){
      $functions->{$func}{netcdf} =~ s/nattsp/ngattsp/g;
  }

  if(defined ($functions->{$func}{pnetcdf}) && defined($functions->{$func}{netcdf})){
#      && defined($functions->{$func}{pio})){
      my $buf=0;
      foreach my $line (@tmptmp){
	  if($line =~ /msg = 0/){
	      my $msg = "PIO_MSG_".$func;
	      $msg =~ tr/a-z/A-Z/;
	      $line =~ s/msg = 0/msg = $msg/;
#	      print "  $msg,\n";
	  }
	  if($line =~/function/){
	      if($line =~ s/PIO_function/PIOc_$func/){
		  $line =~ s/\(\)/ $functions->{$func}{pnetcdf} / ; 
#	      $line =~ s/int ncid/file_desc_t *file/;
		  $line =~ s/MPI_Offset/PIO_Offset/g;
		  $line =~ s/\;//;
		  
	      }else{
		  my $args;
		  if($line =~ s/ncmpi_function/ncmpi_$func/){
		      $args = $functions->{$func}{pnetcdf} ;
		  }
		  if($line =~ s/nc_function/nc_$func/){
		      $args = $functions->{$func}{netcdf}  ; 
		      if($functions->{$func}{pnetcdf} =~ /void \*buf/  ){
			  $args =~ s/\*op/ buf/g;
			  $args =~ s/\*ip/ buf/g;
		      }
		      if($functions->{$func}{pnetcdf} =~ /\*ip/  ){
			  $args =~ s/\*op/ ip/g;
		      }
		  }
		  $args =~ s/int ncid/file->fh/;
#      $args =~ s/int varid/vdesc->varid/;
		  $args =~ s/const //g;
		  $args =~ s/unsigned //g;
		  $args =~ s/signed //g;
		  $args =~ s/char //g;
		  $args =~ s/nc_type //g;
		  $args =~ s/MPI_Offset //g;
		  $args =~ s/MPI_Datatype //g;
		  $args =~ s/int //g;
		  $args =~ s/short //g;
		  $args =~ s/long //g;
		  $args =~ s/float //g;
		  $args =~ s/double //g;
		  $args =~ s/void//g;
		  $args =~ s/MPI_Info //g;
		  $args =~ s/,\s+\*/, /g;
		  $args =~ s/\[\]//g;
		  
		  $args =~ s/size_t \*indexp/\(size_t \*\) index/g;
		  $args =~ s/size_t \*startp/\(size_t \*\) start/g;
		  $args =~ s/size_t \*countp/\(size_t \*\) count/g;
		  $args =~ s/ptrdiff_t \*stridep/\(ptrdiff_t \*\) stride/g;
		  $args =~ s/ptrdiff_t \*\s*imapp/\(ptrdiff_t \*\) imap/g;


#		  $args =~ s/ptrdiff_t \*/\(ptrdiff_t \*\)/g;
#		  if($args =~ s/size_t \*/\(size_t \*\)/g){
#		  }else{
#		      $args =~ s/size_t /\(size_t\)/g;
#		  }
		  
		  $line =~ s/\(\)/$args/;
	      }
	  }
#	  if($line =~ /chkerr =/ && $func =~ /def_var/){
#	      print F "  file->varlist[*varidp].record=-1;\n";
#	      print F "  file->varlist[*varidp].buffer=NULL;\n";
#	  }
	  print F $line;
      }
    
#    print "$func  $functions->{$func}{pnetcdf} $functions->{$func}{netcdf}\n";
      print F "\n";
  }
}
close(F);
