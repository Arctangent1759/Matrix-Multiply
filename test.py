#!/usr/bin/python

import sys

def parseline(x):
  """parses a line. duh."""
  return float(x[x.find(" \t "):].strip()[:x[x.find(" \t "):].strip().find(" ")]);
def classify(x):
  out={
	'classification':'unknown',
	'pass':False,
	'rows':0,
	'cols':0,
	'speed':0,
  }
  out['rows']=float(x[:x.find("m")].strip().split(" ")[0]);
  out['cols']=float(x[:x.find("m")].strip().split(" ")[2]);
  out['speed']=parseline(x);
  if out['cols']>out['rows']:
	out['classification']='wide';
  elif out['cols']<out['rows']:
	out['classification']='long';
  else:
	out['classification']='square';
  out['pass']=(out['speed']>=5.0);
  return out;

def computeAvg(s):
  if num[s]==0:
	return "Not a Number"
  else:
	return float(total[s])/float(num[s]);

total={
	'long':0,
	'wide':0,
	'square':0,
	'all':0,
}
num={
	'long':0,
	'wide':0,
	'square':0,
	'all':0,
}
lists={
	'long':[],
	'wide':[],
	'square':[],
	'all':[],
}
while (True):
  line_in=sys.stdin.readline().strip();
  try:
	curr=classify(line_in);
  except ValueError:
	break; #end of loop
  print curr;

  lists[curr['classification']].append(curr);
  total[curr['classification']]+=curr['speed'];
  num[curr['classification']]+=1;

  lists['all'].append(curr);
  total['all']+=curr['speed'];
  num['all']+=1;

  print("""Averages:
  long: {0}
  wide: {1}
  square: {2}
  all: {3}
 """.format(
 computeAvg('long'),
 computeAvg('wide'),
 computeAvg('square'),
 computeAvg('all')));
