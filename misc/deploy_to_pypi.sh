#!/bin/bash

rm -r build dist
mv platon/data .
python3 setup.py sdist bdist_wheel
twine upload dist/*
mv data platon
