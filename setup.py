# -*- coding: utf-8 -*-
from setuptools import find_packages, setup
from os import path as os_path
import time
this_directory = os_path.abspath(os_path.dirname(__file__))
"""帮助[https://www.notion.so/6bade2c6a5f4479f82a4e67eafcebb3a]

上传到anaconda
https://docs.anaconda.com/anacondaorg/user-guide/tasks/work-with-packages/
 
    """
# 读取文件内容
def read_file(filename):
    with open(os_path.join(this_directory, filename), encoding='utf-8') as f:
        long_description = f.read()
    return long_description

# 获取依赖
def read_requirements(filename):
    return [line.strip() for line in read_file(filename).splitlines()
            if not line.startswith('#')]
long_description="""

一个分类模型


"""

long_description=read_file("README.md")
setup(
    name='tkitClassifierLit', #修改包名字-
    version='0.0.0.1.4',
    description='一个分类模型',
    author='Terry Chan',
    author_email='napoler2008@gmail.com',
    url='https://github.com/napoler/tkitClassifierLit',
    # install_requires=read_requirements('requirements.txt'),  # 指定需要安装的依赖
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        'pytorch-lightning==1.2.7',
        'transformers==4.5.0'

    ],
    packages=['tkitClassifierLit'])

"""
pip freeze > requirements.txt

python3 setup.py sdist
#python3 setup.py install
python3 setup.py sdist upload
"""
