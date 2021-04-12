
#encoding=utf-8
from __future__ import unicode_literals
import sys
# 切换到上级目录
sys.path.append("../")
# 引入本地库
import tkitClassifierLit

cl =tkitClassifierLit.ClassifierLit()
model=cl()

print(model)
