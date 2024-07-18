#!/usr/bin/env python3

import sys
import os
from dotenv import load_dotenv
import argparse
import getpass

import pypdf
import langchain
import langchain_core
import langchain_community
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate



def example_function(argument):
    return f"Received argument: {argument}"


def main(example_arg=None):
    if example_arg:
        print(example_function(example_arg))
    else:
        print("Hello, world!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Description of your program.")
    parser.add_argument('--example', type=str, help='An example argument')
    args = parser.parse_args()

    main(args.example)



