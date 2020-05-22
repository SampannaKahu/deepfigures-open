#!/bin/bash -x

RESULTS_DIR=$1

rm -rf $RESULTS_DIR/arxiv_data_output/diffs_100dpi/*
rm -rf $RESULTS_DIR/arxiv_data_output/src/*
rm -rf $RESULTS_DIR/arxiv_data_output/modified_src/*
rm -rf $RESULTS_DIR/arxiv_data_output/figure-jsons/*
rm -rf $RESULTS_DIR/arxiv_data_temp/*

