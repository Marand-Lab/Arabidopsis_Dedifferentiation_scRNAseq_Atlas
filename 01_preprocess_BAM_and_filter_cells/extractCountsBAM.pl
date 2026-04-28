#!/usr/bin/perl
use strict;
use warnings;
use Sort::Naturally;

# usage
die "perl $0 <file.v2.bam>\n" unless @ARGV == 1;

# vars
my %meta;
my %cnts;
my $library;
my $it = 0;

# run
open F, "samtools view $ARGV[0] | " or die;
while(<F>){
	chomp;
	$it++;
	if(($it % 1000000) == 0){
		print STDERR " - iterated over $it records ...\n";
	}
	my @col = split("\t",$_);
	my $bc;
	my $trx;
	my $lib;
	for (my $i = 11; $i < @col; $i++){
		if($col[$i] =~ /^CB:Z:/){
			$bc = $col[$i];
		}elsif($col[$i] =~ /^RG:Z:/){
			my @info = split(":",$col[$i]);
			$lib = $info[2];
		}elsif($col[$i] =~ /^GX:Z:/){
			my @info = split(":",$col[$i]);
			$trx = $info[2];
		}
		if($bc && $lib && $trx){
			last;
		}
	}
	if($bc){

		# update global var lib with library name
		if(!$library && $lib){
			$library = $lib;
		}

		# update bc id
		#$bc = substr($bc, 0, (length($bc)-1));
		$bc = $bc . "-" . $lib;
		
		# initialize bc
		if(!exists $meta{$bc}){
			$meta{$bc}{'total'} = 0;
			$meta{$bc}{'trx'} = 0;
			$meta{$bc}{'nuclear'} = 0;
			$meta{$bc}{'ChrC'} = 0;
			$meta{$bc}{'ChrM'} = 0;
		}

		# update metadata
		$meta{$bc}{'total'}++;
		if($trx){
			$meta{$bc}{'trx'}++;
		}

		# check for organellar alignments
		if($col[2] eq 'ChrC'){
			$meta{$bc}{'ChrC'}++;
		}elsif($col[2] eq 'ChrM'){
			$meta{$bc}{'ChrM'}++;
		}else{
			$meta{$bc}{'nuclear'}++;
		}
		
		# update counts
		if($trx){
			$cnts{$bc}{$trx}++;
		}
	}
}
close F;

# print metadata
my $metaout = $library . ".metadata.txt";
open (my $mo, '>', $metaout) or die;
print $mo "cellID\ttotal\ttrx\tnuclear\tPt\tMt\tlibrary\n";
my @barcodes = keys %cnts;
for (my $i = 0; $i < @barcodes; $i++){
	print $mo "$barcodes[$i]\t$barcodes[$i]\t$meta{$barcodes[$i]}{'total'}\t$meta{$barcodes[$i]}{'trx'}\t$meta{$barcodes[$i]}{'nuclear'}\t$meta{$barcodes[$i]}{'ChrC'}\t$meta{$barcodes[$i]}{'ChrM'}\t$library\n";
}
close $mo;

# print sparse counts
my $countout = $library . ".sparse";
open (my $co, '>', $countout) or die;
for (my $i = 0; $i < @barcodes; $i++){
	my @genes = nsort keys %{$cnts{$barcodes[$i]}};
	for (my $j = 0; $j < @genes; $j++){
		print $co "$genes[$j]\t$barcodes[$i]\t$cnts{$barcodes[$i]}{$genes[$j]}\n";
	}
}
close $co;

