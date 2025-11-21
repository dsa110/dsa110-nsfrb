#!/bin/bash

systemctl --user stop procserver_search
systemctl --user stop T4manager.service
systemctl --user stop procserver_RX
systemctl --user stop rt_injector_test.service
