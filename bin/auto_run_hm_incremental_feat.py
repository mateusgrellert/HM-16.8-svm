
from subprocess import check_output, PIPE, STDOUT
from sys import argv
import os
import re
from Bjontegaard import *

train_files = os.listdir(argv[1])
num_frames = 60
rangePath = argv[2]
qps = ['22','27','32','37']
#qps = [32]
re_bitrate = '%d\s*a\s*(\d+.\d+)\s*' % (num_frames)
re_psnr = '%d\s*a\s*\d+.\d+\s*(\d+.\d+)\s*(\d+.\d+)\s*(\d+.\d+)\s*(\d+.\d+)\s*' % (num_frames)
re_time = 'Total\sTime:\s*(\d+.\d+)\s*'

refBDResults = []
refTimes =[]
fout = open('incremental_bd_results_BQMall_%dfr.csv' % (num_frames),'w')
try:
	log = open('svm_log_BQMall_%dfr.log' % (num_frames),'r')
	log_lines = log.readlines()
	log_table = {}
	for seq in sequences:
		for line in log_lines:
			if len (line) < 4: continue
			[seq, name,qp,t,br,y,u,v,yuv] = line.split()
			if seq not in log_table.keys():
				if name not in log_table[seq].keys():
					log_table[seq][name] = {}
				log_table[seq][name][qp] = [t,br,y,u,v,yuv]

	log.close()
except:
	log_table = {}

log = open('svm_log_%dfr.log' % (num_frames),'a')

for qp in qps:
	cmd = './TAppEncoderStatic -c ../cfg/encoder_lowdelay_P_main.cfg -c ~/hm-cfgs/BQMall.cfg -f %d -q %s --UseSVM=0 ' % (num_frames, qp)
	if 'ref' in log_table.keys():
		if qp in log_table['ref'].keys():
			[refTime, bitrate, y,u,v, yuv] = log_table['ref'][qp]
			print 'REF Time qp %s: %s BR: %s YUV: %s' % (qp,refTime, bitrate, yuv)
			refBDResults.append([float(bitrate),float(yuv)])
			refTimes.append(float(refTime))

			continue

	output = check_output(cmd, shell=True,stderr=PIPE)
	bitrate = re.search(re_bitrate, output).group(1)
	refTime = re.search(re_time,output).group(1)
	refTimes.append(float(refTime))
	[y,u,v,yuv] = re.search(re_psnr,output).groups()
	print 'REF Time qp %s: %s BR: %s YUV: %s' % (qp,refTime, bitrate, yuv)
	print >> log, '\t'.join([str(x) for x in ['ref',qp,refTime, bitrate, y,u,v,yuv]])
	refBDResults.append([float(bitrate),float(yuv)])


for trf in train_files:
	if ('train' not in trf) or ('qp22' not in trf): continue

	fail = False
	testBDResults = []
	avg_tr = 0.0
	for qp in qps:
		ttrf = trf.replace('qp22','qp'+str(qp))


		if ttrf in log_table.keys():
			if qp in log_table[ttrf].keys():
				[time,bitrate, y,u,v, yuv] = log_table[ttrf][qp]
				testBDResults.append([float(bitrate),float(yuv)])

				tr = float(time)/float(refTimes[qps.index(qp)])
				avg_tr += tr
				print 'TRAIN: %s Time: %.2f TimeRatio: %.2f BR: %s YUV: %s' % (ttrf, float(time), tr, bitrate, yuv)

				continue

		cmd = './TAppEncoderStatic -c ../cfg/encoder_lowdelay_P_main.cfg -c ~/hm-cfgs/BQMall.cfg -f %d -q %s --UseSVM=1 --SvmModelPath=%s%s --SvmScaleRangePath=%s ' % (num_frames, qp,argv[1],ttrf, rangePath)
		#print cmd
		try:
			output = check_output(cmd, shell=True,stderr=PIPE)
			bitrate = re.search(re_bitrate, output).group(1)
			time = re.search(re_time,output).group(1)
			[y,u,v,yuv] = re.search(re_psnr,output).groups()
			testBDResults.append([float(bitrate),float(yuv)])

			tr = float(time)/float(refTimes[qps.index(qp)])
			avg_tr += tr
			print 'TRAIN: %s Time: %.2f TimeRatio: %.2f BR: %s YUV: %s' % (ttrf, float(time), tr, bitrate, yuv)
			print >> log, '\t'.join([str(x) for x in [ttrf,qp,time, bitrate, y,u,v,yuv]])

		except:
			fail = True
	if not fail:
		bdr = bdrate(refBDResults, testBDResults)
		print >> fout, ttrf, '\t', bdr, '\t', avg_tr/len(qps)
		print '\t', ttrf, '\t', bdr, '\t', avg_tr/len(qps)
	else:
		print 'Could not get results for %s' % (ttrf)


log.close()
fout.close()