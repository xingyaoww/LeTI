#!/bin/bash
export PYTHONPATH=`pwd`:$PYTHONPATH
gunicorn leti.verifier.eae.api_server:app -b localhost:5000
