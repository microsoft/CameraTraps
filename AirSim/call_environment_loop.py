import subprocess
import airsim

import pprint
import os
import time
import itertools
import psutil
import random
import json

from get_camtrap_images import get_camtrap_images

environment_file = 'environment_lookup.json'

def kill(proc_pid):
    process = psutil.Process(proc_pid)
    for proc in process.children(recursive=True):
        proc.kill()
    process.kill()

with open(environment_file, 'r') as f:
    environment_lookup = json.load(f)
#environment_lookup = {}
last_env_collected = len(environment_lookup.keys())


biome_seed = [0] #int
biome_density = [30.0, 60.0, 90.0] #float 0-100
biome_class_density_trees = [10.0, 25.0, 50.0, 75.0] #float 0-100
biome_class_density_boulders = [25.0, 50.0, 75.0] #float 0-100
biome_class_density_rocks = [25.0, 50.0, 75.0] #float 0-100
biome_class_density_shrubs = [25.0, 50.0, 75.0] #float 0-100
biome_class_density_grass = [25.0, 50.0, 75.0] #float 0-100
biome_class_density_logs = [25.0, 50.0, 75.0] #float 0-100
animal_class = [0, 3, 4, 5, 6, 9]#list(range(10)) #list of ints 0-9
animal_seed = [0] #int
animal_density = [1, 3, 5] #int

env_list = itertools.product(biome_seed,biome_density,biome_class_density_trees, 
    biome_class_density_boulders,biome_class_density_rocks,biome_class_density_shrubs, 
    biome_class_density_grass, biome_class_density_logs, animal_class,animal_seed, animal_density)
#print(len(list(env_list)))
for idx, env in enumerate(env_list):
    env_num = idx+last_env_collected
    environment_lookup[env_num] = {'BiomeSeed':random.randint(1,1000),
                                    'BiomeDensity':env[1],
                                    'BiomeClassDensityTrees': env[2],
                                    'BiomeClassDensityBoulders': env[3],
                                    'BiomeClassDensityRocks': env[4],
                                    'BiomeClassDensityShrubs': env[5],
                                    'BiomeClassDensityGrass': env[6],
                                    'BiomeClassDensityLogs': env[7],
                                    'AnimalClass': env[8],
                                    'AnimalSeed': random.randint(1,1000),
                                    'AnimalDensity': env[10]}
    proc = subprocess.Popen(['C:\\Users\\t-sabeer\\Documents\\AirSim\\TrapCam\\Binaries - Rev2\\WindowsNoEditor\\TrapCam.exe',
        '-BiomeSeed',str(environment_lookup[env_num]['BiomeSeed']),
        '-BiomeDensity',str(environment_lookup[env_num]['BiomeDensity']),
        '-BiomeClassDensityTrees',str(environment_lookup[env_num]['BiomeClassDensityTrees']),
        '-BiomeClassDensityBoulders',str(environment_lookup[env_num]['BiomeClassDensityBoulders']),
        '-BiomeClassDensityRocks',str(environment_lookup[env_num]['BiomeClassDensityRocks']),
        '-BiomeClassDensityShrubs',str(environment_lookup[env_num]['BiomeClassDensityShrubs']),
        '-BiomeClassDensityGrass',str(environment_lookup[env_num]['BiomeClassDensityGrass']),
        '-BiomeClassDensityLogs',str(environment_lookup[env_num]['BiomeClassDensityLogs']),
        '-AnimalClass', str(environment_lookup[env_num]['AnimalClass']),
        '-AnimalSeed', str(environment_lookup[env_num]['AnimalSeed']),
        '-AnimalDensity', str(environment_lookup[env_num]['AnimalDensity'])])

    print('Running app, environment '+ str(env_num))
    time.sleep(4)
    get_camtrap_images(env_num, environment_lookup[env_num]['AnimalClass'],environment_lookup[env_num]['AnimalDensity'])
    print('Images collected')
    time.sleep(2)
    #save environment dict every time so you don't lose the info if airsim crashes
    with open(environment_file,'w') as f:
        json.dump(environment_lookup, f)

    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        kill(proc.pid)

with open(environment_file,'w') as f:
    json.dump(environment_lookup, f)
