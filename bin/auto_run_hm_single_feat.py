
from subprocess import check_output, PIPE, STDOUT
from sys import argv
import os
import re
from Bjontegaard import *

train_files = os.listdir(argv[1])
num_frames = 60
rangePath = argv[2]
qps = [22,27,32,37]
qps = [32]
re_bitrate = '%d\s*a\s*(\d+.\d+)\s*' % (num_frames)
re_psnr = '%d\s*a\s*\d+.\d+\s*(\d+.\d+)\s*(\d+.\d+)\s*(\d+.\d+)\s*(\d+.\d+)\s*' % (num_frames)
re_time = 'Total\sTime:\s*(\d+.\d+)\s*'

refBDResults = []
refTimes =[]
fout = open('single_bd_results_BQMall_%dfr.csv' % (num_frames),'w')

for qp in qps:
	cmd = './TAppEncoderStatic -c ../cfg/encoder_lowdelay_P_main.cfg -c ~/hm-cfgs/BQMall.cfg -f %d -q %d --UseSVM=0 ' % (num_frames, qp)
	output = check_output(cmd, shell=True,stderr=PIPE)
	bitrate = re.search(re_bitrate, output).group(1)
	refTime = re.search(re_time,output).group(1)
	refTimes.append(float(refTime))
	[y,u,v,yuv] = re.search(re_psnr,output).groups()
	print 'REF Time: %s BR: %s YUV: %s' % (refTime, bitrate, yuv)

	refBDResults.append([float(bitrate),float(yuv)])


for trf in train_files:
	if 'train' not in trf: continue


	fail = False
	testBDResults = []
	avg_tr = 0.0
	for qp in qps:
		cmd = './TAppEncoderStatic -c ../cfg/encoder_lowdelay_P_main.cfg -c ~/hm-cfgs/BQMall.cfg -f %d -q %d --UseSVM=1 --SvmModelPath=%s%s --SvmScaleRangePath=%s ' % (num_frames, qp,argv[1],trf, rangePath)
		#print cmd
		if 1:
			output = check_output(cmd, shell=True,stderr=PIPE)
			bitrate = re.search(re_bitrate, output).group(1)
			time = re.search(re_time,output).group(1)
			[y,u,v,yuv] = re.search(re_psnr,output).groups()
			testBDResults.append([float(bitrate),float(yuv)])

			tr = float(time)/float(refTimes[qps.index(qp)])
			avg_tr += tr
			print 'TRAIN: %s Time: %.2f TimeRatio: %.2f BR: %s YUV: %s' % (trf, float(time), tr, bitrate, yuv)
		else:
			fail = True
	"""if not fail:
		bdr = bdrate(refBDResults, testBDResults)
		print >> fout, trf, '\t', bdr, '\t', avg_tr/len(qps)
	else:
		print 'Could not get results for %s' % (trf)"""

