from subprocess import call
from string import Template


jobs_spec= []
with open("jobs_info_both.txt") as f:
  keys=[]
  for i,line in enumerate(f.readlines()):
    if i==0:
      keys= line[:-1].split(",")
    else:
      if not line.startswith("#"):
        values= line[:-1].split(",")
        print(i,keys,values,len(keys), len(values))
        assert len(keys)==len(values), "something is missing"
        tmp_dict={}
        for k in range(0,len(keys)):
          tmp_dict[keys[k]]=values[k]
        jobs_spec.append(tmp_dict)
print(jobs_spec)

#open the file
filein = open( 'job_template.tmpl' )
#read it
src = Template( filein.read() )
#do the substitution
for dict in jobs_spec:
  with open(dict['name']+".sh", "w") as text_file:
    text_file.write(src.substitute(dict))
  print(call("sbatch_dgx "+dict['name']+".sh", shell=True))
