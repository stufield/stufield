# Linux Cheat Sheet of (Ubuntu) Commands

Stu Field

20 September 2024

-------------------------


## Symbolic Links

```bash
ln -sf [target] [location]         # soft link
ln -Lf [target] [location]         # hard link
```

## VIM
### rebuild vim spellfile

```bash
:mkspell! ~/.vim/spell/en.utf-8.add
```

### Multiple buffer search/replace
```bash
:bufdo %s/<pattern>/<replace>/ge | update
```


## Check System Status CRON

```bash
systemctl status cron
```


## Finding

```bash
find [path] -name [*pattern*]       # find filenames matching the quote string from current root
grep -r [pattern] [file]            # find a search string inside files and list them
     -i   find a search string inside files and list them (case insensitive)
     -l    find a search string inside files and list only the files that contain them
grep [pattern] $(find [path] -type f -name [file pattern])   # grep for string only in certain files from find
grep [pattern] **/*.R               # via Z-shell
```

- open in VIM as buffers
```bash
vi $(grep -lr [pattern] *)
vi $(grep -lr [pattern] `find -name *.R`)
vi $(grep -l [pattern] **/*.R)   # using Z-shell
```


## Mounting drives

```bash
sudo mount -t cifs -o "username=sfield,password=*********,uid=sfield,gid=sfield" //aspen/shared /media/S
sudo mount -t cifs -o "username=sfield,password=*********,uid=sfield,gid=sfield" //aspen/res_dev /media/R
sudo mount -t cifs -o "username=sfield,password=*********,uid=sfield,gid=sfield" //aspen/home/sfield /media/H
```


## BASH tricks

```bash
cd -  OR cd $OLDPWD               # go back to previous directory
touch blahfile{1,2,3}.txt         # create multiple files
mv blahfile{1,one}.txt            # rename file with old/new name
mkdir -p test/{shintaro,stu}      # make subdirectories
mkdir -p test2/f1/f2              # make parents as necessary (linear tree)
!!                                # repeat last command
!                                 # repeat last search command
rm !(*.c|*.py)                    # remove all files except those with .c or .py extension
rm ./-file                        # remove file beginning with hyphen
echo $?                           # return value of last command
!^                                # first argument
!$                                # last argument
!:[number]                        # number argument.
# moves filename1 to destination
# example
touch filename1 filename2
mkdir destination
mv !touch^ !$
```


## Trailing/Leading Spaces

```bash
gsub("[[:space:]]*$", "", x)
gsub("^[[:space:]]*", "", x)
```


## Trailing Tabs

```bash
:%s/\s\+$//
```


## Use grouping grep

```bash
gsub("^(.+)(SL[0-9]+_[0-9][a-z]).+$", "\\2", "data/CalibratorReference.SL14386_9f.txt")
```


## chmod + chown

```bash
u = user
g = group
o = others
a = all
r = read permission
w = write permission
x = execute permission
# -------------------------------- #
chmod 775 *.R                     change permissions for all files with the .R extension
chmod u+x filename                give user permission to execute
chmod u+x,g+x filename            give user & group permission to execute
chmod a+x filename                give everybody execute permissions
chmod u-rx filename               remove read and excute permissions for user
chmod --reference=file1 file2     match permissions of file2 to file1
chmod -R dirname/                 apply permissions to all files under directory recursively
chown sfield:sfield file          change ownership permissions
# -------------------------------- #
7 = rwx
6 = rw-
5 = r-x
4 = r--
3 = -wx
2 = -w-
1 = --x
0 = ---
```


## Lists

```bash
du -sch *          size of directory
ls -l | wc          size and number of visible files in current directory
ls -a | wc          size and number of ALL files in current directory
ls -l | wc -l       how many visible files in current directory
ls -a | wc -l       how many TOTAL files in current directry
ls -alS             list all files by size
ls -dl filename     get permissions for a file
```


## Trash

```bash
~/.local/share/Trash/files
```


## Zip

```bash
zip filename.zip file1 file2 file3 ...
zip -e filename.zip file1 file2 file3 ...  (with password)
unzip filename.zip                         unzip file
unzip filename.zip -d dirname              unzip file to a directory
unzip -l filename.zip                      list contents of zip file
unzip -l filename.zip | vi -               list contents of zip file to vi
```


## Tar

```bash
tar -cvzf filename.tar.gz file1 file2 file3 ...   tar first, then zip
tar -cvf filename.tar file1 file2 file3  ...      just tarball
tar -xvf filename.tar                             untar to current directory
tar -xvf filename.tar -C dirname/                 untar to specified directory
gunzip filename.tar.gz                            unzip the tarball, create tarball
tar -tvf filename.tar                             list contents of tarball
tar -ztvf filename.tar.gz                         list contents of zipped tarball
```


# Replacing text in files
# based on search pattern

```bash
for i in *.txt
  do sed -e "s/string/replace/" $i > fixed/$i
done
```


## Running processes

```bash
ps -ef | grep "string"
```


# German Esset and Umlaut characters

```bash
Crtl + Shift + u + 00df + <Enter>
RightAlt + ss       -> ß     (in sequence; must set RightAlt to Alternative Characters Key in Settings)
RightAlt + u + "    -> ü
RightAlt + a + "    -> ä
```


## pushd

```bash
dirs                 list the directory stack
dirs -p              list the directory stack, each on a new line
dirs -c              clear the directory stack
pushd dir            add directory to the stack and change to that dir
pushd -n dir         add directory to the stack and suppress dir change
pushd *no args*      switch 1st and 2nd dirs in stack
pushd -N             Rearrange the stack, bringing the Nth stack entry to front, rotating the rest of the stack
pushd +N             Same above, but the Nth entry from the left (above is Nth entry from right).
popd                 pop 1st dir off the stack and switch to 2nd
popd +N              remove the Nth stack entry from the top
popd -N              remove the Nth stack entry from the bottom
```


## Kill a Process

```bash
top
ps -ef | grep exec
kill -USR1 <#>
```


## Manage PDFs

```bash
pdftk *.pdf cat output combined.pdf               # combine multiple PDFs
pdftk combined.pdf burst output mypage_%02d.pdf   # separate multi-page PDF into single PDFs
```


## cron settings

```bash
   * * * * *      command to execute
   | | | | |
   | | | | |
   | | | | +----   day of week (0 - 6) | Sunday=0
   | | | +------   month of year (1 - 12)
   | | +--------   day of month (1 - 31)
   | +----------   hour of day (0 - 23)
   +------------   min of hour (0 - 59)
```

## get drive/mount disk space

```bash
df -Th
```


## Compile R and install from source

```bash
mv ~/Downloads/R-3.1.0.tar.gz ~/Apps         # move tarball to dest dir
tar -zvxf R-3.1.0.tar.gz                     # unzip
cd R-3.1.0                                   # go into unzipped dir
#./configure --prefix=/usr                   # run configure for /usr
./configure                                  # run configure
make                                         # run make to finish local install
make check                                   # checks that system-wide install will work
#sudo make install                           # installs R to root in /usr/local/
#sudo make uninstall                         # uninstall R
```


## run R examples check only

```bash
R -q --vanilla < SomaPkg-Ex.R     # this is run on the Ex file generated by R CMD check
```


## VirtualBox

```bash
deb http://download.virtualbox.org/virtualbox/debian wily contrib      # virtualbox in sources.list
https://www.virtualbox.org/wiki/Linux_Downloads
sudo apt-get install virtualbox virtualbox-dkms
sudo apt-get install virtualbox-guest-additions-iso
# Oracle_VM_VirtualBox_Extension_Pack-5.0.14-105127.vbox-extpack
Install extension pack by: File -> Preferences -> Extensions
                           Remove old extensions and add new
```


## Batch Rename

```bash
rename 's/pattern/replace/g' [pattern].log      # regular log-text files
for i in *.tar.gz
  do svn mv $i `echo $i | sed 's/4/5/'`    # files under SVN control
done      
```


## Git get code from deleted file

```bash
git log --all -- <path-to-file>
git show <SHA> -- <path-to-file>
```



## Update Java

```bash
sudo apt-get install oracle-java8-installer
sudo apt-get install oracle-java8-set-default
java --version
```


## Ubuntu Release

```bash
lsb_release -a
```


## apt-get

```bash
sudo apt-get install linux-headers-generic build-essential
sudo vi /etc/apt/sources.list
sudo apt-get update
apt-cache showpkg r-base
sudo apt-get install <pkgs>
```


## Increase VBox HD Storage

```bash
VBoxManage modifyhd --compact   # this should release memory from the VM is dynamically taken
VBoxManage modifyhd --resize 80000 VMwin7.vdi   # 80GB
File -> Virtual Media Manager -> Release then Remove
Settings -> Storage -> Controller:SATA -> Add Icon -> "point to *.vdi" -> OK
```


