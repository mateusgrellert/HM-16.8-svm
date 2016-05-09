
from subprocess import check_output, PIPE, STDOUT
from sys import argv
import os
import re

train_files = os.listdir(argv[1])
num_frames = 20

for trf in train_files:
	if 'train' not in trf: continue

	cmd = './TAppEncoderStatic -c ../cfg/encoder_lowdelay_P_main.cfg -c ~/hm-cfgs/BQMall.cfg -f %d --UseSVM=1 --SvmModelPath=%s%s ' % (num_frames,argv[1],trf)
	re_bitrate = '%d\s*a\s*(\d+.\d+)\s*' % (num_frames)
	re_psnr = '%d\s*a\s*\d+.\d+\s*(\d+.\d+)\s*(\d+.\d+)\s*(\d+.\d+)\s*(\d+.\d+)\s*' % (num_frames)
	re_time = 'Total\sTime:\s*(\d+.\d+)\s*'
	try:
		output = check_output(cmd, shell=True,stderr=PIPE)
		bitrate = re.search(re_bitrate, output).group(1)
		time = re.search(re_time,output).group(1)
		[y,u,v,yuv] = re.search(re_psnr,output).groups()
		print 'TRAIN: %s Time: %s BR: %s YUV: %s' % (trf, time, bitrate, yuv)
	except:
		print 'Could not get results for %s' % (trf)
