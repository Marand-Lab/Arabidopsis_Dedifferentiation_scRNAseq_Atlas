#!/usr/bin/perl
use strict;
use warnings;

open F, $ARGV[0] or die;
while(<F>){
	if($_ =~ /;/){
		chomp;
		my @col = split("\t",$_);
		my @genes = split(";",$col[0]);
		foreach(@genes){
			print "$_\t$col[1]\t$col[2]\n";
		}
	}else{
		print "$_";
	}
}
close F;
