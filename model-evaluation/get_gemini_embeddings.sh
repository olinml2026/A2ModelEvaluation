#!/bin/bash

curl -X POST \
     -H "Authorization: Bearer $(gcloud auth print-access-token)" \
     -H "Content-Type: application/json; charset=utf-8" \
     -d @request.json \
     "https://us-central1-aiplatform.googleapis.com/v1/projects/yourprojecthere/locations/us-central1/publishers/google/models/gemini-embedding-001:predict" > gemini_embeddings.json
