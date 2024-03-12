#!/usr/bin/env bash

# get kaldi network in text form from training output
kaldi_text=convert.kaldi.txt

# make a copy of kwsbp resource, 
# add all keywords in json format into keywords.json
# query 'threshold1' according to det statistics file
# revise 'threshold1' for each keyword in keywords.json 
in_resource=kwsbp_resource

# set an output path
out_dir=output

out_resource=$out_dir/kwsbp_resource
out_resource_quant16=$out_dir/kwsbp_resource_quant16
out_resource_quant16_bin=$out_dir/kwsbp_resource_quant16.bin

if [ ! -d $out_dir ]; then
    mkdir -p $out_dir
fi

export PATH=./:$PATH
export LD_LIBRARY_PATH=./:$LD_LIBRARY_PATH
chmod -R 755 ./

# 1. convert to kaldi bin format
./nnet-copy --binary=true $kaldi_text $out_dir/convert_kaldi.net

# 2. copy kwsbp resource
if [ ! -d $out_resource ]; then
    mkdir -p $out_resource
fi
cp -r $in_resource/* $out_resource/
cp $out_dir/convert_kaldi.net $out_resource/kwsr.net

# 3. quant network to 16bit
./quanter nnet1 $out_resource/kwsr ${out_resource}.kwsr.quant16 16bit
if [ ! -d $out_resource_quant16 ]; then
    mkdir -p $out_resource_quant16
fi
cp -r $out_resource/* $out_resource_quant16/
mv ${out_resource}.kwsr.quant16.mdl $out_resource_quant16/kwsr.net

# 4. pack all kwsbp resource to bin file
if [ -f $out_resource_quant16_bin ]; then
  rm -f $out_resource_quant16_bin
fi
./packer -pd $out_resource_quant16 $out_resource_quant16_bin
