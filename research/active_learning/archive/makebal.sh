
for d in */ ; do  
  echo $d
  find $d -type f -print0 | sort -zR | tail -zn +101 |xargs -0 rm
done

