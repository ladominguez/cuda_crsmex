#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 22 15:11:17 2018

@author: antonio
"""

from obspy.core import read 


a = read('20110106064638.IG.CAIG.HHZ.sac')

a.plot()