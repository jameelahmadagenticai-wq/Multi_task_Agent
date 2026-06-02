#!/bin/bash

docker exec -it ai_ollama ollama pull phi3:mini

docker exec -it ai_ollama ollama pull mistral:7b

docker exec -it ai_ollama ollama pull llava:7b

docker exec -it ai_ollama ollama pull deepseek-r1:7b