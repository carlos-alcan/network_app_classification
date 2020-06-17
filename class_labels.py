#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 12:12:23 2020

@author: carlosalcantara
"""

'''
Loads flow data aggregated with application labels using ndpi into a pandas DataFrame, adds class labels according to 
specifications in the replace function. Creates original file with new csv with class labels as "savefile". Note that
"csvfile" is overwritten if savefile is not specified.

Usage: class_labels.py csvfile [savefile=csvfile]
'''

import pandas as pd
import sys

# Check for command line argument
if len(sys.argv) < 2:
    print('Usage: class_labels.py csvfile [savefile=csvfile]')
    sys.exit(-1)

# if savefile not specified, overwrite csvfile
file = sys.argv[1]
if len(sys.argv) > 2:
    savefile = sys.argv[2]
else:
    savefile = file

# read csv file
df = pd.read_csv(file)
# remove zero duration flows
df = df[df.duration > 0]
    
# add class label column to df with app label
df['class'] = df['app']

# replace ndpi application label with its corresponding class
#  - add or remove rows here to edit application to class mappings
df = df.replace({'class': {'Amazon'          :   'Big Tech', 
                           'Apple'           :   'Big Tech', 
                           'AppleiCloud'     :   'unknown', 
                           'AppleiTunes'     :   'unknown', 
                           'ApplePush'       :   'M2M Messaging',  
                           'AppleStore'       :  'Big Tech',  
                           'BGP'             :   'Network Operation', 
                           'BitTorrent'      :   'File Transfer',  
                           'COAP'            :   'unknown', 
                           'Cloudflare'      :   'Network Operation', 
                           'CNN'             :   'News/Information', 
                           'DHCPV6'          :   'Network Operation', 
                           'DNS'             :   'Network Operation', 
                           'FTP_CONTROL'     :   'File Transfer', 
                           'FTP_DATA'        :   'File Transfer', 
                           'Facebook'        :   'unknown', 
                           'GMail'           :   'unknown', 
#                           'Github'          :   '', 
                           'Google'          :   'Big Tech', 
                           'GoogleDocs'      :   'unknown', 
                           'GoogleDrive'     :   'unknown', 
                           'GoogleHangout'   :   'Chat_VoIP',  
                           'GoogleMaps'      :   'unknown', 
                           'GoogleServices'  :   'M2M Messaging',  
#                           'HTTP'            :   '', 
                           'ICMP'            :   'Network Operation', 
                           'ICMPV6'          :   'Network Operation', 
                           'IGMP'            :   'Network Operation', 
                           'IRC'             :   'Chat_VoIP', 
                           'Kerberos'        :   'Authentication', 
                           'LinkedIn'        :   'unknown', 
                           'MDNS'            :   'Network Operation', 
                           'MQTT'            :   'M2M Messaging', 
                           'MS_OneDrive'     :   'unknown', 
                           'Microsoft'       :   'Big Tech', 
                           'MSN'             :   'News/Information', 
                           'NFS'             :   'File Transfer', 
                           'NTP'             :   'Network Operation',  
                           'NetBIOS'         :   'unknown', 
                           'Office365'       :   'unknown', 
                           'Oscar'           :   'Chat_VoIP', 
                           'PlayStore'       :   'Big Tech', 
                           'QQ'              :   'Chat_VoIP', 
                           'QUIC'            :   'Chat_VoIP', 
                           'RDP'             :   'Remote login', 
                           'RTMP'            :   'Chat_VoIP', 
                           'Redis'           :   'unknown', 
                           'RX'              :   'unknown', 
                           'SIP'             :   'Chat_VoIP', 
                           'SNMP'            :   'Network Management', 
                           'SSDP'            :   'Network Operation', 
                           'SSH'             :   'Remote login', 
                           'SSL'             :   'HTTPS',
                           'SSL_No_Cert'     :   'HTTPS', 
                           'STUN'            :   'Chat_VoIP',  
                           'Skype'           :   'Chat_VoIP',  
                           'SkypeCall'       :   'Chat_VoIP', 
                           'Slack'           :   'Chat_VoIP', 
                           'Syslog'          :   'Network Management', 
                           'TeamSpeak'       :   'Chat_VoIP', 
                           'Teredo'          :   'Network Operation', 
                           'Tor'             :   'unknown', 
                           'Twitter'         :   'unknown', 
                           'UPnP'            :   'Network Operation',
                           'UbuntuONE'       :   'Big Tech', 
                           'Viber'           :   'Chat_VoIP', 
                           'Whois-DAS'       :   'Network Management', 
                           'Wikipedia'       :   'News/Information', 
                           'Yahoo'           :   'Big Tech', 
                           'YouTube'         :   'Video Streaming', 
#                           'unknown'         :   '', 
                              }})

df.to_csv(savefile, index=False)
