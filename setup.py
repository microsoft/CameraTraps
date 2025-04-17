from setuptools import setup, find_packages

with open('README.md', encoding="utf8") as file:
        long_description = file.read()
setup(
    name='PytorchWildlife',
    version='1.2.1', 
    packages=find_packages(),
    url='https://github.com/microsoft/CameraTraps/',  
    license='MIT',
    author='Andres Hernandez, Zhongqi Miao, Daniela Ruiz Lopez, Isai Daniel Chacon Silva',
    author_email='v-herandres@microsoft.com, zhongqimiao@microsoft.com, v-druizlopez@microsoft.com, v-ichaconsil@microsoft.com',  
    description='a PyTorch Collaborative Deep Learning Framework for Conservation.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=[
        'torch',
        'torchvision',
        'torchaudio',
        'tqdm',
        'Pillow', 
        'supervision==0.23.0',
        'gradio',
        'ultralytics',
        'chardet',
        'wget',
        'yolov5',
        'setuptools',
        'scikit-learn',
        'timm',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',  
        'Intended Audience :: Developers', 
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    keywords='pytorch_wildlife, pytorch, wildlife, megadetector, conservation, animal, detection, classification',
    python_requires='>=3.8',
)
