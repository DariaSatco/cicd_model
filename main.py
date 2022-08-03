import os

if __name__ == '__main__':

    # pull data with DVC
    if "DYNO" in os.environ and os.path.isdir(".dvc"):
        os.system("dvc config core.no_scm true")
        if os.system("dvc pull") != 0:
            exit("dvc pull failed")
        os.system("rm -r .dvc .apt/usr/lib/dvc")