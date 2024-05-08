# encoding: utf-8
import time
import logging
import itertools
import requests
import json
import multiprocessing as mp

#from semantic_search_en.database import Database
#from semantic_search_en.api_model import Model

file_list=['A','B','C','D','E','F','G','H']

#file_list=['H']
url = "http://localhost:8899/compute/semantic_srch_en/"
header = {"X-Patsnap-Version": "v1"}



def batching(xfiles, n):
    minibatches = []
    l = len(xfiles)
    for ndx in range(0, l, n):
        minibatches.append(xfiles[ndx:min(ndx + n, l)])
    return minibatches


def load_pids(filename):
    pids_list = []
    with open(filename) as xfile:
        for json_line in xfile:
            json_dict = json.loads(json_line)
            pid = json_dict["pid"]
            pids_list.append(pid)

    return pids_list

def eval_single_batch(minibatch):
    results_list=[]

    for pid in minibatch:
        pbdt='20181010'
        header['X-Correlation-ID'] = pid
        search_query={"data": { "patent_id": pid, "date_ranges": [{ "field": "PBD", "begin": "*", "end": pbdt}], "rows": 501} }
        for i in range(15):
            try:
                rep = requests.post(url, json=search_query, headers=header)
                if rep.status_code == 200:
                    result = rep.json()['data']['query_response']['response']['docs']
                    result_dict={"pid":pid,"pbdt":pbdt,"results":result}
                    results_list.append(result_dict)
            except json.decoder.JSONDecodeError:
                logging.warning("Error:"+str(pid))
                time.sleep(1)
                continue

            except KeyError:
                logging.warning("Key Error:"+str(pid))
                continue

            except requests.exceptions.ChunkedEncodingError:
                logging.warning("Chunk Error:"+str(pid))
                time.sleep(1)
                continue
            break
    return results_list

def eval_all_reports(xfiles):
    pool = mp.Pool(20)
    reports = batching(xfiles, 8)
    results = pool.map(eval_single_batch, reports)

    results = list(itertools.chain.from_iterable(results))
    return results