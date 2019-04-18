import json
import sqlite3

def get_type(val):
    if isinstance(val,str):
      return "TEXT";
    elif isinstance(val,int):
      return "INTEGER"
    elif isinstance(val,float):
      return "REAL"
    else:
      print("Unknown Type Error")
      raise

def create_sql(name, dic):
    fields= ['X','Y','WIDTH','HEIGHT']
    keys = ''
    for the_key, the_value in dic.items():
      if not isinstance(the_value, list):
        keys+= ","+str(the_key)+" "+str(get_type(the_value))
      else:
        for i,v in enumerate(the_value):
          keys+= ","+str(the_key)+"_"+fields[i]+" "+get_type(v)
    return 'CREATE TABLE '+name+' ('+keys[1:]+')'

def insert_sql(name, dic):
    keys = ','.join(dic.keys())
    extra=0
    if keys.find("bbox")>=0:
      keys=keys.replace("bbox","bbox_X,bbox_Y,bbox_WIDTH,bbox_HEIGHT")
      extra=3
    question_marks = ','.join(list('?'*(len(dic)+extra)))
    return 'INSERT INTO '+name+' ('+keys+') VALUES ('+question_marks+')'

def prepare_params(vals):
  params= []
  for val in vals:
    if not isinstance(val,list):
      params.append(val)
    else:
      if len(val)==0:
        params.extend([None,None,None,None])
      else:
        for v in val:
          params.append(v)
  #print(params)
  return tuple(params)

with open("mnt/databases/emammal/eMammal_20180929.json") as f:
  loaded_json = json.loads(f.read())
  conn=sqlite3.connect("emammal.db")

  for key in loaded_json.keys():
    print(key)
    records= loaded_json[key]
    if isinstance(records,list):
      sample= records[0]
    else:
      sample= records
      #print(type(sample),sample)
    cur= conn.cursor()
    create_query= create_sql(key, sample)
    insert_query= insert_sql(key, sample)
    #print(insert_query)
    cur.execute(create_query)

    for record in records:
      if isinstance(record,dict):
        cur.execute(insert_query,prepare_params(record.values()))
      else:
        cur.execute(insert_query,prepare_params(records))
        break
    cur.close()
    conn.commit()
    
