import sys, os, distutils.core, subprocess

if not os.path.exists('./detectron2'):
    subprocess.run(['git', 'clone', 'https://github.com/facebookresearch/detectron2'])

dist = distutils.core.run_setup("./detectron2/setup.py")

for x in dist.install_requires:
    subprocess.run(['python', '-m', 'pip', 'install', x])

sys.path.insert(0, os.path.abspath('./detectron2'))