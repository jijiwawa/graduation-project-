# -*- coding:utf-8 -*-
from distutils.core import setup, Extension

MOD = 'Extest' #模块名
setup(name=MOD,ext_modules=[Extension(MOD,sources=['my_extend.c'])]) #源文件名