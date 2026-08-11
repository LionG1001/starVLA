#!/bin/bash
pod=${1:-blake-precheck}
kubectl get po -nhis-test -owide |grep $pod |grep Running
