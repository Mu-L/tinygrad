#!/usr/bin/env python3
from ane import ANE
ane = ANE()

lens = {}
pos = {}

dat = b"\xff"*0x300
ret = ane.debug(dat, 16)
for k,v in ret.items():
  found = None
  for i in range(33):
    #print(v, (1 << i) - 1)
    if v == (1 << i) - 1:
      found = i
      break
  #print(k, hex(v), found)
  lens[k] = found

dat = b"\x00"*0x300
for i in range(0x300):
  for j in range(8):
    dat = b"\x00"*i
    dat += bytes([1 << j])
    dat += b"\x00"*(0x300-len(dat))
    ret = ane.debug(dat, 16)
    for k,v in ret.items():
      if v == 1:
        print("0x%3x %d %2d" % (i, j, lens[k]), k)
        pos[k] = (i,j, lens[k])

print(len(lens), len(pos))

