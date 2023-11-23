from setuptools import setup, find_packages

setup(
    name='PytorchWildlife',
    version='1.0.0', 
    packages=find_packages(),
    url='https://github.com/microsoft/CameraTraps/tree/PytorchWildlife_Dev',  
    license='MIT',
    author='Andres Hernandez, Zhongqi Miao',
    author_email='v-andreshern@microsoft.com, zhongqimiao@microsoft.com',  
    description='a PyTorch Collaborative Deep Learning Framework for Conservation.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=[
        'numpy',
        'torch==1.10.1',
        'torchvision==0.11.2',
        'torchaudio==0.10.1',
        'tqdm==4.66.1',
        'Pillow==10.1.0', 
        'supervision==0.16.0',
        'gradio',
        'ultralytics-yolov5'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',  
        'Intended Audience :: Developers', 
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    keywords='pytorch, wildlife, megadetector, conservation, animal, detection, classification',
    python_requires='>=3.8',
)
