import glob
import os

def main(pardir):

    dirlist = sorted(glob.glob("%s/*_BEAGLE_*"%pardir))

    for i in range(26,94):
        for file in dirlist:
            x = file.split('/')[-1]
            if int(x.split('_')[0])>1e5 and x[:2]=="%d"%i:
                x = "%d"%(i-1) + x[2:]
                newfile = "%s/%s"%(pardir,x)
                os.system("mv %s %s"%(file,newfile))

if __name__ == '__main__':

    main(pardir="results/const/")
    main(pardir="results/errfix/")
    main(pardir="results/galsub/")
