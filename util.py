import os
import time
import hashlib
import progressbar
from PIL import Image

def loadLables(path):
    f = open(path, 'rb')
    label_lines = f.readlines()
    f.close()
    return label_lines

def checkFolder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def formatSec(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return "%d:%02d:%02d" % (h, m, s)

def logAndWriteFile(f, msg, printMsg = True):
    if f : f.write(msg + "\n")
    if printMsg : print msg

PIL_IMG_FORMAT_JPEG = "JPEG"
PIL_IMG_FORMAT_PNG = "PNG"
def ensureImageFormatValid(path, validImageSet = [PIL_IMG_FORMAT_JPEG, PIL_IMG_FORMAT_PNG], delInvaidImageFile = False):
    def __delImg(path):
        if delInvaidImageFile:
            os.remove(img_path)
            print "Delete '%s' OK" % img_path

    start = time.time()
    count = 0
    for dirPath, dirNames, fileNames in os.walk(path):
        print "Scanning %d files from %s ... " % (len(fileNames), dirPath)
        for f in fileNames:
            img_path = os.path.join(dirPath, f)
            count += 1
            try:
                im = Image.open(img_path)
                if im.format not in validImageSet:
                    print 'Not desired image fotmat:', im.format, img_path
                    __delImg(img_path)
            except Exception as err:
                print "[ERROR]", err
                __delImg(img_path)

    print "Complete, %d images scanned, spend %f sec" % (count, (time.time() - start))

def getLabelDisplayString(label_lines):
    maxLabelLen = 0
    for l in label_lines:
        if len(l) > maxLabelLen:
            maxLabelLen = len(l)

    return "%%%ds" % maxLabelLen

def randomMoveFiles(src, count, dst):
    '''
    Random move files from src folder to dst folder
    '''
    lstFiles = os.listdir(src)
    if not os.path.exists(dst):
        os.makedirs(dst)
    import random
    random.shuffle(lstFiles)
    lstFiles = lstFiles[:count]
    for f in lstFiles:
        srcFile = os.path.join(src, f)
        dstFile = os.path.join(dst, f)
        os.rename(srcFile, dstFile)
    print "move %d files complete" % len(lstFiles)

def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def removeDuplicatedImage(src):
    start = time.time()
    bar = progressbar.ProgressBar()
    lstFiles = os.listdir(src)
    lstMd5 = []
    count = 0
    for f in bar(lstFiles):
        md5v = md5(os.path.join(src, f))
        if md5v in lstMd5:
            print "Remove duplicated file : %s" % f
            count += 1
            os.remove(os.path.join(src, f))
        else:
            lstMd5.append(md5v)
    if count > 0:
        print "Removed %d files @ %s" % (count, src)
    print "Complete, %d images scanned, spend %f sec" % (len(lstFiles), (time.time() - start))

if __name__ == '__main__':
    src= "/data/deep_learning/dataset/training/pycon_2017_demo/unknown"
    count = 200
    dst = "/data/deep_learning/dataset/test/pycon_2017_demo/unknown"
    randomMoveFiles(src, count, dst)

if __name__ == '__main__2':
    root = "/data/deep_learning/dataset/test/pg_label_set"

    removeDuplicatedImage(root)

    lstDirs = os.listdir(root)
    bar = progressbar.ProgressBar()
    for d in bar(lstDirs):
        print " >>>>>>>>>>>>>>>>> Processing : %s" % d
        removeDuplicatedImage(os.path.join(root, d))
        ensureImageFormatValid(os.path.join(root, d), validImageSet = [PIL_IMG_FORMAT_JPEG], delInvaidImageFile = True)
