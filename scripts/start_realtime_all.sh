#!/bin/bash

systemctl --user start procserver_search
sleep 30
systemctl --user start T4manager.service
systemctl --user start procserver_RX
systemctl --user start rt_injector_test.service
