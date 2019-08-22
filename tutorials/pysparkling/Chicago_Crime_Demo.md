# 1. Define Spark Context

`sc`

`<pyspark.context.SparkContext at 0x102cea1d0>`


# 2. Start H2O Context
    
```
    
    from pysparkling import *
    sc
    hc= H2OContext(sc).start()

    Warning: Version mismatch. H2O is version 3.6.0.2, but the python package is version 3.7.0.99999.

```

<div style="overflow:auto"><table style="width:50%"><tr><td>H2O cluster uptime: </td>
<td>2 seconds 217 milliseconds </td></tr>
<tr><td>H2O cluster version: </td>
<td>3.6.0.2</td></tr>
<tr><td>H2O cluster name: </td>
<td>sparkling-water-nidhimehta</td></tr>
<tr><td>H2O cluster total nodes: </td>
<td>2</td></tr>
<tr><td>H2O cluster total memory: </td>
<td>3.83 GB</td></tr>
<tr><td>H2O cluster total cores: </td>
<td>16</td></tr>
<tr><td>H2O cluster allowed cores: </td>
<td>16</td></tr>
<tr><td>H2O cluster healthy: </td>
<td>True</td></tr>
<tr><td>H2O Connection ip: </td>
<td>172.16.2.98</td></tr>
<tr><td>H2O Connection port: </td>
<td>54329</td></tr></table></div>



# 3. Define H2O Context

```
    hc

    H2OContext: ip=172.16.2.98, port=54329
```

# 4. Import H2O Python library
    
    import h2o
    
# 5. View all available H2O Python functions
    
    #dir(h2o)

# 6. Parse Chicago Crime dataset into H2O
    column_type = ['Numeric','String','String','Enum','Enum','Enum','Enum','Enum','Enum','Enum','Numeric','Numeric','Numeric','Numeric','Enum','Numeric','Numeric','Numeric','Enum','Numeric','Numeric','Enum']
    f_crimes = h2o.import_file(path ="../data/chicagoCrimes10k.csv",col_types =column_type)
    
    print(f_crimes.shape)
    f_crimes.summary()

    
    Parse Progress: [##################################################] 100%
    (9999, 22)



<table>
<tr><th>       </th><th>ID           </th><th>Case Number  </th><th>Date                  </th><th>Block               </th><th>IUCR  </th><th>Primary Type   </th><th>Description                 </th><th>Location Description  </th><th>Arrest        </th><th>Domestic      </th><th>Beat         </th><th>District     </th><th>Ward         </th><th>Community Area  </th><th>FBI Code  </th><th>X Coordinate  </th><th>Y Coordinate  </th><th>Year  </th><th>Updated On            </th><th>Latitude       </th><th>Longitude      </th><th>Location                     </th></tr>
<tr><td>type   </td><td>int          </td><td>string       </td><td>string                </td><td>enum                </td><td>enum  </td><td>enum           </td><td>enum                        </td><td>enum                  </td><td>enum          </td><td>enum          </td><td>int          </td><td>int          </td><td>int          </td><td>int             </td><td>enum      </td><td>int           </td><td>int           </td><td>int   </td><td>enum                  </td><td>real           </td><td>real           </td><td>enum                         </td></tr>
<tr><td>mins   </td><td>21735.0      </td><td>NaN          </td><td>NaN                   </td><td>0.0                 </td><td>0.0   </td><td>0.0            </td><td>0.0                         </td><td>0.0                   </td><td>0.0           </td><td>0.0           </td><td>111.0        </td><td>1.0          </td><td>1.0          </td><td>1.0             </td><td>0.0       </td><td>1100317.0     </td><td>1814255.0     </td><td>2015.0</td><td>0.0                   </td><td>41.64507243    </td><td>-87.906463888  </td><td>0.0                          </td></tr>
<tr><td>mean   </td><td>9931318.73737</td><td>NaN          </td><td>NaN                   </td><td>NaN                 </td><td>NaN   </td><td>NaN            </td><td>NaN                         </td><td>NaN                   </td><td>0.292829282928</td><td>0.152315231523</td><td>1159.61806181</td><td>11.3489885128</td><td>22.9540954095</td><td>37.4476447645   </td><td>NaN       </td><td>1163880.59815 </td><td>1885916.14984 </td><td>2015.0</td><td>NaN                   </td><td>41.8425652247  </td><td>-87.6741405221 </td><td>NaN                          </td></tr>
<tr><td>maxs   </td><td>9962898.0    </td><td>NaN          </td><td>NaN                   </td><td>6517.0              </td><td>212.0 </td><td>26.0           </td><td>198.0                       </td><td>90.0                  </td><td>1.0           </td><td>1.0           </td><td>2535.0       </td><td>25.0         </td><td>50.0         </td><td>77.0            </td><td>24.0      </td><td>1205069.0     </td><td>1951533.0     </td><td>2015.0</td><td>32.0                  </td><td>42.022646183   </td><td>-87.524773286  </td><td>8603.0                       </td></tr>
<tr><td>sigma  </td><td>396787.564221</td><td>NaN          </td><td>NaN                   </td><td>NaN                 </td><td>NaN   </td><td>NaN            </td><td>NaN                         </td><td>NaN                   </td><td>0.455083515588</td><td>0.35934414686 </td><td>695.76029875 </td><td>6.94547493301</td><td>13.6495661144</td><td>21.2748762223   </td><td>NaN       </td><td>16496.4493681 </td><td>31274.0163199 </td><td>0.0   </td><td>NaN                   </td><td>0.0860186579358</td><td>0.0600357970653</td><td>NaN                          </td></tr>
<tr><td>zeros  </td><td>0            </td><td>0            </td><td>0                     </td><td>3                   </td><td>16    </td><td>11             </td><td>933                         </td><td>19                    </td><td>7071          </td><td>8476          </td><td>0            </td><td>0            </td><td>0            </td><td>0               </td><td>16        </td><td>0             </td><td>0             </td><td>0     </td><td>603                   </td><td>0              </td><td>0              </td><td>1                            </td></tr>
<tr><td>missing</td><td>0            </td><td>0            </td><td>0                     </td><td>0                   </td><td>0     </td><td>0              </td><td>0                           </td><td>6                     </td><td>0             </td><td>0             </td><td>0            </td><td>162          </td><td>0            </td><td>0               </td><td>0         </td><td>162           </td><td>162           </td><td>0     </td><td>0                     </td><td>162            </td><td>162            </td><td>162                          </td></tr>
<tr><td>0      </td><td>9955810.0    </td><td>HY144797     </td><td>02/08/2015 11:43:40 PM</td><td>081XX S COLES AVE   </td><td>1811  </td><td>NARCOTICS      </td><td>POSS: CANNABIS 30GMS OR LESS</td><td>STREET                </td><td>true          </td><td>false         </td><td>422.0        </td><td>4.0          </td><td>7.0          </td><td>46.0            </td><td>18        </td><td>1198273.0     </td><td>1851626.0     </td><td>2015.0</td><td>02/15/2015 12:43:39 PM</td><td>41.747693646   </td><td>-87.549035389  </td><td>(41.747693646, -87.549035389)</td></tr>
<tr><td>1      </td><td>9955861.0    </td><td>HY144838     </td><td>02/08/2015 11:41:42 PM</td><td>118XX S STATE ST    </td><td>0486  </td><td>BATTERY        </td><td>DOMESTIC BATTERY SIMPLE     </td><td>APARTMENT             </td><td>true          </td><td>true          </td><td>522.0        </td><td>5.0          </td><td>34.0         </td><td>53.0            </td><td>08B       </td><td>1178335.0     </td><td>1826581.0     </td><td>2015.0</td><td>02/15/2015 12:43:39 PM</td><td>41.679442289   </td><td>-87.622850758  </td><td>(41.679442289, -87.622850758)</td></tr>
<tr><td>2      </td><td>9955801.0    </td><td>HY144779     </td><td>02/08/2015 11:30:22 PM</td><td>002XX S LARAMIE AVE </td><td>2026  </td><td>NARCOTICS      </td><td>POSS: PCP                   </td><td>SIDEWALK              </td><td>true          </td><td>false         </td><td>1522.0       </td><td>15.0         </td><td>29.0         </td><td>25.0            </td><td>18        </td><td>1141717.0     </td><td>1898581.0     </td><td>2015.0</td><td>02/15/2015 12:43:39 PM</td><td>41.87777333    </td><td>-87.755117993  </td><td>(41.87777333, -87.755117993) </td></tr>
<tr><td>3      </td><td>9956197.0    </td><td>HY144787     </td><td>02/08/2015 11:30:23 PM</td><td>006XX E 67TH ST     </td><td>1811  </td><td>NARCOTICS      </td><td>POSS: CANNABIS 30GMS OR LESS</td><td>STREET                </td><td>true          </td><td>false         </td><td>321.0        </td><td>nan          </td><td>6.0          </td><td>42.0            </td><td>18        </td><td>nan           </td><td>nan           </td><td>2015.0</td><td>02/15/2015 12:43:39 PM</td><td>nan            </td><td>nan            </td><td>                             </td></tr>
<tr><td>4      </td><td>9955846.0    </td><td>HY144829     </td><td>02/08/2015 11:30:58 PM</td><td>0000X S MAYFIELD AVE</td><td>0610  </td><td>BURGLARY       </td><td>FORCIBLE ENTRY              </td><td>APARTMENT             </td><td>false         </td><td>false         </td><td>1513.0       </td><td>15.0         </td><td>29.0         </td><td>25.0            </td><td>05        </td><td>1137239.0     </td><td>1899372.0     </td><td>2015.0</td><td>02/15/2015 12:43:39 PM</td><td>41.880025548   </td><td>-87.771541324  </td><td>(41.880025548, -87.771541324)</td></tr>
<tr><td>5      </td><td>9955835.0    </td><td>HY144778     </td><td>02/08/2015 11:30:21 PM</td><td>010XX W 48TH ST     </td><td>0486  </td><td>BATTERY        </td><td>DOMESTIC BATTERY SIMPLE     </td><td>APARTMENT             </td><td>false         </td><td>true          </td><td>933.0        </td><td>9.0          </td><td>3.0          </td><td>61.0            </td><td>08B       </td><td>1169986.0     </td><td>1873019.0     </td><td>2015.0</td><td>02/15/2015 12:43:39 PM</td><td>41.807059405   </td><td>-87.65206589   </td><td>(41.807059405, -87.65206589) </td></tr>
<tr><td>6      </td><td>9955872.0    </td><td>HY144822     </td><td>02/08/2015 11:27:24 PM</td><td>015XX W ARTHUR AVE  </td><td>1320  </td><td>CRIMINAL DAMAGE</td><td>TO VEHICLE                  </td><td>STREET                </td><td>false         </td><td>false         </td><td>2432.0       </td><td>24.0         </td><td>40.0         </td><td>1.0             </td><td>14        </td><td>1164732.0     </td><td>1943222.0     </td><td>2015.0</td><td>02/15/2015 12:43:39 PM</td><td>41.999814056   </td><td>-87.669342967  </td><td>(41.999814056, -87.669342967)</td></tr>
<tr><td>7      </td><td>21752.0      </td><td>HY144738     </td><td>02/08/2015 11:26:12 PM</td><td>060XX W GRAND AVE   </td><td>0110  </td><td>HOMICIDE       </td><td>FIRST DEGREE MURDER         </td><td>STREET                </td><td>true          </td><td>false         </td><td>2512.0       </td><td>25.0         </td><td>37.0         </td><td>19.0            </td><td>01A       </td><td>1135910.0     </td><td>1914206.0     </td><td>2015.0</td><td>02/15/2015 12:43:39 PM</td><td>41.920755683   </td><td>-87.776067514  </td><td>(41.920755683, -87.776067514)</td></tr>
<tr><td>8      </td><td>9955808.0    </td><td>HY144775     </td><td>02/08/2015 11:20:33 PM</td><td>001XX W WACKER DR   </td><td>0460  </td><td>BATTERY        </td><td>SIMPLE                      </td><td>OTHER                 </td><td>false         </td><td>false         </td><td>122.0        </td><td>1.0          </td><td>42.0         </td><td>32.0            </td><td>08B       </td><td>1175384.0     </td><td>1902088.0     </td><td>2015.0</td><td>02/15/2015 12:43:39 PM</td><td>41.886707818   </td><td>-87.631396356  </td><td>(41.886707818, -87.631396356)</td></tr>
<tr><td>9      </td><td>9958275.0    </td><td>HY146732     </td><td>02/08/2015 11:15:36 PM</td><td>001XX W WACKER DR   </td><td>0460  </td><td>BATTERY        </td><td>SIMPLE                      </td><td>HOTEL/MOTEL           </td><td>false         </td><td>false         </td><td>122.0        </td><td>1.0          </td><td>42.0         </td><td>32.0            </td><td>08B       </td><td>1175384.0     </td><td>1902088.0     </td><td>2015.0</td><td>02/15/2015 12:43:39 PM</td><td>41.886707818   </td><td>-87.631396356  </td><td>(41.886707818, -87.631396356)</td></tr>
</table>



# 7. Look at the distribution of the IUCR column
    f_crimes["IUCR"].table()


<table>
<tr><th>IUCR  </th><th style="text-align: right;">  Count</th></tr>
<tr><td>0110  </td><td style="text-align: right;">     16</td></tr>
<tr><td>0261  </td><td style="text-align: right;">      2</td></tr>
<tr><td>0263  </td><td style="text-align: right;">      2</td></tr>
<tr><td>0265  </td><td style="text-align: right;">      5</td></tr>
<tr><td>0266  </td><td style="text-align: right;">      2</td></tr>
<tr><td>0281  </td><td style="text-align: right;">     41</td></tr>
<tr><td>0291  </td><td style="text-align: right;">      3</td></tr>
<tr><td>0312  </td><td style="text-align: right;">     18</td></tr>
<tr><td>0313  </td><td style="text-align: right;">     20</td></tr>
<tr><td>031A  </td><td style="text-align: right;">    136</td></tr>
</table>


#8. Look at the distribution of the Arrest column
    f_crimes["Arrest"].table()


<table>
<tr><th>Arrest  </th><th style="text-align: right;">  Count</th></tr>
<tr><td>false   </td><td style="text-align: right;">   7071</td></tr>
<tr><td>true    </td><td style="text-align: right;">   2928</td></tr>
</table>

# 9. Modify column names to replace blank spaces with underscores
    col_names = map(lambda s: s.replace(' ', '_'), f_crimes.col_names)
    f_crimes.set_names(col_names)


<table>
<tr><th style="text-align: right;">             ID</th><th>Case_Number  </th><th>Date                  </th><th>Block               </th><th style="text-align: right;">  IUCR</th><th>Primary_Type   </th><th>Description                 </th><th>Location_Description  </th><th>Arrest  </th><th>Domestic  </th><th style="text-align: right;">  Beat</th><th style="text-align: right;">  District</th><th style="text-align: right;">  Ward</th><th style="text-align: right;">  Community_Area</th><th>FBI_Code  </th><th style="text-align: right;">  X_Coordinate</th><th style="text-align: right;">  Y_Coordinate</th><th style="text-align: right;">  Year</th><th>Updated_On            </th><th style="text-align: right;">  Latitude</th><th style="text-align: right;">  Longitude</th><th>Location                     </th></tr>
<tr><td style="text-align: right;">    9.95581e+06</td><td>HY144797     </td><td>02/08/2015 11:43:40 PM</td><td>081XX S COLES AVE   </td><td style="text-align: right;">  1811</td><td>NARCOTICS      </td><td>POSS: CANNABIS 30GMS OR LESS</td><td>STREET                </td><td>true    </td><td>false     </td><td style="text-align: right;">   422</td><td style="text-align: right;">         4</td><td style="text-align: right;">     7</td><td style="text-align: right;">              46</td><td>18        </td><td style="text-align: right;">   1.19827e+06</td><td style="text-align: right;">   1.85163e+06</td><td style="text-align: right;">  2015</td><td>02/15/2015 12:43:39 PM</td><td style="text-align: right;">   41.7477</td><td style="text-align: right;">   -87.549 </td><td>(41.747693646, -87.549035389)</td></tr>
<tr><td style="text-align: right;">    9.95586e+06</td><td>HY144838     </td><td>02/08/2015 11:41:42 PM</td><td>118XX S STATE ST    </td><td style="text-align: right;">  0486</td><td>BATTERY        </td><td>DOMESTIC BATTERY SIMPLE     </td><td>APARTMENT             </td><td>true    </td><td>true      </td><td style="text-align: right;">   522</td><td style="text-align: right;">         5</td><td style="text-align: right;">    34</td><td style="text-align: right;">              53</td><td>08B       </td><td style="text-align: right;">   1.17834e+06</td><td style="text-align: right;">   1.82658e+06</td><td style="text-align: right;">  2015</td><td>02/15/2015 12:43:39 PM</td><td style="text-align: right;">   41.6794</td><td style="text-align: right;">   -87.6229</td><td>(41.679442289, -87.622850758)</td></tr>
<tr><td style="text-align: right;">    9.9558e+06 </td><td>HY144779     </td><td>02/08/2015 11:30:22 PM</td><td>002XX S LARAMIE AVE </td><td style="text-align: right;">  2026</td><td>NARCOTICS      </td><td>POSS: PCP                   </td><td>SIDEWALK              </td><td>true    </td><td>false     </td><td style="text-align: right;">  1522</td><td style="text-align: right;">        15</td><td style="text-align: right;">    29</td><td style="text-align: right;">              25</td><td>18        </td><td style="text-align: right;">   1.14172e+06</td><td style="text-align: right;">   1.89858e+06</td><td style="text-align: right;">  2015</td><td>02/15/2015 12:43:39 PM</td><td style="text-align: right;">   41.8778</td><td style="text-align: right;">   -87.7551</td><td>(41.87777333, -87.755117993) </td></tr>
<tr><td style="text-align: right;">    9.9562e+06 </td><td>HY144787     </td><td>02/08/2015 11:30:23 PM</td><td>006XX E 67TH ST     </td><td style="text-align: right;">  1811</td><td>NARCOTICS      </td><td>POSS: CANNABIS 30GMS OR LESS</td><td>STREET                </td><td>true    </td><td>false     </td><td style="text-align: right;">   321</td><td style="text-align: right;">       nan</td><td style="text-align: right;">     6</td><td style="text-align: right;">              42</td><td>18        </td><td style="text-align: right;"> nan          </td><td style="text-align: right;"> nan          </td><td style="text-align: right;">  2015</td><td>02/15/2015 12:43:39 PM</td><td style="text-align: right;">  nan     </td><td style="text-align: right;">   nan     </td><td>                             </td></tr>
<tr><td style="text-align: right;">    9.95585e+06</td><td>HY144829     </td><td>02/08/2015 11:30:58 PM</td><td>0000X S MAYFIELD AVE</td><td style="text-align: right;">  0610</td><td>BURGLARY       </td><td>FORCIBLE ENTRY              </td><td>APARTMENT             </td><td>false   </td><td>false     </td><td style="text-align: right;">  1513</td><td style="text-align: right;">        15</td><td style="text-align: right;">    29</td><td style="text-align: right;">              25</td><td>05        </td><td style="text-align: right;">   1.13724e+06</td><td style="text-align: right;">   1.89937e+06</td><td style="text-align: right;">  2015</td><td>02/15/2015 12:43:39 PM</td><td style="text-align: right;">   41.88  </td><td style="text-align: right;">   -87.7715</td><td>(41.880025548, -87.771541324)</td></tr>
<tr><td style="text-align: right;">    9.95584e+06</td><td>HY144778     </td><td>02/08/2015 11:30:21 PM</td><td>010XX W 48TH ST     </td><td style="text-align: right;">  0486</td><td>BATTERY        </td><td>DOMESTIC BATTERY SIMPLE     </td><td>APARTMENT             </td><td>false   </td><td>true      </td><td style="text-align: right;">   933</td><td style="text-align: right;">         9</td><td style="text-align: right;">     3</td><td style="text-align: right;">              61</td><td>08B       </td><td style="text-align: right;">   1.16999e+06</td><td style="text-align: right;">   1.87302e+06</td><td style="text-align: right;">  2015</td><td>02/15/2015 12:43:39 PM</td><td style="text-align: right;">   41.8071</td><td style="text-align: right;">   -87.6521</td><td>(41.807059405, -87.65206589) </td></tr>
<tr><td style="text-align: right;">    9.95587e+06</td><td>HY144822     </td><td>02/08/2015 11:27:24 PM</td><td>015XX W ARTHUR AVE  </td><td style="text-align: right;">  1320</td><td>CRIMINAL DAMAGE</td><td>TO VEHICLE                  </td><td>STREET                </td><td>false   </td><td>false     </td><td style="text-align: right;">  2432</td><td style="text-align: right;">        24</td><td style="text-align: right;">    40</td><td style="text-align: right;">               1</td><td>14        </td><td style="text-align: right;">   1.16473e+06</td><td style="text-align: right;">   1.94322e+06</td><td style="text-align: right;">  2015</td><td>02/15/2015 12:43:39 PM</td><td style="text-align: right;">   41.9998</td><td style="text-align: right;">   -87.6693</td><td>(41.999814056, -87.669342967)</td></tr>
<tr><td style="text-align: right;">21752          </td><td>HY144738     </td><td>02/08/2015 11:26:12 PM</td><td>060XX W GRAND AVE   </td><td style="text-align: right;">  0110</td><td>HOMICIDE       </td><td>FIRST DEGREE MURDER         </td><td>STREET                </td><td>true    </td><td>false     </td><td style="text-align: right;">  2512</td><td style="text-align: right;">        25</td><td style="text-align: right;">    37</td><td style="text-align: right;">              19</td><td>01A       </td><td style="text-align: right;">   1.13591e+06</td><td style="text-align: right;">   1.91421e+06</td><td style="text-align: right;">  2015</td><td>02/15/2015 12:43:39 PM</td><td style="text-align: right;">   41.9208</td><td style="text-align: right;">   -87.7761</td><td>(41.920755683, -87.776067514)</td></tr>
<tr><td style="text-align: right;">    9.95581e+06</td><td>HY144775     </td><td>02/08/2015 11:20:33 PM</td><td>001XX W WACKER DR   </td><td style="text-align: right;">  0460</td><td>BATTERY        </td><td>SIMPLE                      </td><td>OTHER                 </td><td>false   </td><td>false     </td><td style="text-align: right;">   122</td><td style="text-align: right;">         1</td><td style="text-align: right;">    42</td><td style="text-align: right;">              32</td><td>08B       </td><td style="text-align: right;">   1.17538e+06</td><td style="text-align: right;">   1.90209e+06</td><td style="text-align: right;">  2015</td><td>02/15/2015 12:43:39 PM</td><td style="text-align: right;">   41.8867</td><td style="text-align: right;">   -87.6314</td><td>(41.886707818, -87.631396356)</td></tr>
<tr><td style="text-align: right;">    9.95828e+06</td><td>HY146732     </td><td>02/08/2015 11:15:36 PM</td><td>001XX W WACKER DR   </td><td style="text-align: right;">  0460</td><td>BATTERY        </td><td>SIMPLE                      </td><td>HOTEL/MOTEL           </td><td>false   </td><td>false     </td><td style="text-align: right;">   122</td><td style="text-align: right;">         1</td><td style="text-align: right;">    42</td><td style="text-align: right;">              32</td><td>08B       </td><td style="text-align: right;">   1.17538e+06</td><td style="text-align: right;">   1.90209e+06</td><td style="text-align: right;">  2015</td><td>02/15/2015 12:43:39 PM</td><td style="text-align: right;">   41.8867</td><td style="text-align: right;">   -87.6314</td><td>(41.886707818, -87.631396356)</td></tr>
</table>

# 10. Set time zone to UTC for date manipulation
    h2o.set_timezone("Etc/UTC")
    
#11. Refine the date column 
    def refine_date_col(data, col, pattern):
        data[col]         = data[col].as_date(pattern)
        data["Day"]       = data[col].day()
        data["Month"]     = data[col].month()    # Since H2O indexes from 0
        data["Year"]      = data[col].year()
        data["WeekNum"]   = data[col].week()
        data["WeekDay"]   = data[col].dayOfWeek()
        data["HourOfDay"] = data[col].hour()
        
        # Create weekend and season cols
        data["Weekend"] = (data["WeekDay"] == "Sun" or data["WeekDay"] == "Sat").ifelse(1, 0)[0]
        data["Season"] = data["Month"].cut([0, 2, 5, 7, 10, 12], ["Winter", "Spring", "Summer", "Autumn", "Winter"])
        
    refine_date_col(f_crimes, "Date", "%m/%d/%Y %I:%M:%S %p")
    f_crimes = f_crimes.drop("Date")


#12. Parse Census data into H2O
    f_census = h2o.import_file("../data/chicagoCensus.csv",header=1)
    
    ## Update column names in the table
    col_names = map(lambda s: s.strip().replace(' ', '_'), f_census.col_names)
    f_census.set_names(col_names)
    f_census = f_census[1:78,:]
    print(f_census.dim)
    #f_census.summary()

    
    Parse Progress: [##################################################] 100%
    [77, 9]



# 13. Parse Weather data into H2O
    f_weather = h2o.import_file("../data/chicagoAllWeather.csv")
    f_weather = f_weather[1:]
    print(f_weather.dim)
    #f_weather.summary()

    
    Parse Progress: [##################################################] 100%
    [5162, 6]



# 14. Look at all the null entires in the Weather table
    f_weather[f_weather["meanTemp"].isna()]


<table>
<tr><th style="text-align: right;">  month</th><th style="text-align: right;">  day</th><th style="text-align: right;">  year</th><th style="text-align: right;">  maxTemp</th><th style="text-align: right;">  meanTemp</th><th style="text-align: right;">  minTemp</th></tr>
<tr><td style="text-align: right;">      6</td><td style="text-align: right;">   19</td><td style="text-align: right;">  2008</td><td style="text-align: right;">      nan</td><td style="text-align: right;">       nan</td><td style="text-align: right;">      nan</td></tr>
<tr><td style="text-align: right;">      9</td><td style="text-align: right;">   23</td><td style="text-align: right;">  2008</td><td style="text-align: right;">      nan</td><td style="text-align: right;">       nan</td><td style="text-align: right;">      nan</td></tr>
<tr><td style="text-align: right;">      9</td><td style="text-align: right;">   24</td><td style="text-align: right;">  2008</td><td style="text-align: right;">      nan</td><td style="text-align: right;">       nan</td><td style="text-align: right;">      nan</td></tr>
<tr><td style="text-align: right;">      9</td><td style="text-align: right;">   25</td><td style="text-align: right;">  2008</td><td style="text-align: right;">      nan</td><td style="text-align: right;">       nan</td><td style="text-align: right;">      nan</td></tr>
<tr><td style="text-align: right;">      9</td><td style="text-align: right;">   26</td><td style="text-align: right;">  2008</td><td style="text-align: right;">      nan</td><td style="text-align: right;">       nan</td><td style="text-align: right;">      nan</td></tr>
<tr><td style="text-align: right;">      9</td><td style="text-align: right;">   27</td><td style="text-align: right;">  2008</td><td style="text-align: right;">      nan</td><td style="text-align: right;">       nan</td><td style="text-align: right;">      nan</td></tr>
<tr><td style="text-align: right;">      9</td><td style="text-align: right;">   28</td><td style="text-align: right;">  2008</td><td style="text-align: right;">      nan</td><td style="text-align: right;">       nan</td><td style="text-align: right;">      nan</td></tr>
<tr><td style="text-align: right;">      9</td><td style="text-align: right;">   29</td><td style="text-align: right;">  2008</td><td style="text-align: right;">      nan</td><td style="text-align: right;">       nan</td><td style="text-align: right;">      nan</td></tr>
<tr><td style="text-align: right;">      9</td><td style="text-align: right;">   30</td><td style="text-align: right;">  2008</td><td style="text-align: right;">      nan</td><td style="text-align: right;">       nan</td><td style="text-align: right;">      nan</td></tr>
<tr><td style="text-align: right;">      3</td><td style="text-align: right;">    4</td><td style="text-align: right;">  2009</td><td style="text-align: right;">      nan</td><td style="text-align: right;">       nan</td><td style="text-align: right;">      nan</td></tr>
</table>

#15. Look at the help on `as_h2o_frame` 
    hc.as_spark_frame?
    f_weather

    H2OContext: ip=172.16.2.98, port=54329



<table>
<tr><th style="text-align: right;">  month</th><th style="text-align: right;">  day</th><th style="text-align: right;">  year</th><th style="text-align: right;">  maxTemp</th><th style="text-align: right;">  meanTemp</th><th style="text-align: right;">  minTemp</th></tr>
<tr><td style="text-align: right;">      1</td><td style="text-align: right;">    1</td><td style="text-align: right;">  2001</td><td style="text-align: right;">       23</td><td style="text-align: right;">        14</td><td style="text-align: right;">        6</td></tr>
<tr><td style="text-align: right;">      1</td><td style="text-align: right;">    2</td><td style="text-align: right;">  2001</td><td style="text-align: right;">       18</td><td style="text-align: right;">        12</td><td style="text-align: right;">        6</td></tr>
<tr><td style="text-align: right;">      1</td><td style="text-align: right;">    3</td><td style="text-align: right;">  2001</td><td style="text-align: right;">       28</td><td style="text-align: right;">        18</td><td style="text-align: right;">        8</td></tr>
<tr><td style="text-align: right;">      1</td><td style="text-align: right;">    4</td><td style="text-align: right;">  2001</td><td style="text-align: right;">       30</td><td style="text-align: right;">        24</td><td style="text-align: right;">       19</td></tr>
<tr><td style="text-align: right;">      1</td><td style="text-align: right;">    5</td><td style="text-align: right;">  2001</td><td style="text-align: right;">       36</td><td style="text-align: right;">        30</td><td style="text-align: right;">       21</td></tr>
<tr><td style="text-align: right;">      1</td><td style="text-align: right;">    6</td><td style="text-align: right;">  2001</td><td style="text-align: right;">       33</td><td style="text-align: right;">        26</td><td style="text-align: right;">       19</td></tr>
<tr><td style="text-align: right;">      1</td><td style="text-align: right;">    7</td><td style="text-align: right;">  2001</td><td style="text-align: right;">       34</td><td style="text-align: right;">        28</td><td style="text-align: right;">       21</td></tr>
<tr><td style="text-align: right;">      1</td><td style="text-align: right;">    8</td><td style="text-align: right;">  2001</td><td style="text-align: right;">       26</td><td style="text-align: right;">        20</td><td style="text-align: right;">       14</td></tr>
<tr><td style="text-align: right;">      1</td><td style="text-align: right;">    9</td><td style="text-align: right;">  2001</td><td style="text-align: right;">       23</td><td style="text-align: right;">        16</td><td style="text-align: right;">       10</td></tr>
<tr><td style="text-align: right;">      1</td><td style="text-align: right;">   10</td><td style="text-align: right;">  2001</td><td style="text-align: right;">       34</td><td style="text-align: right;">        26</td><td style="text-align: right;">       19</td></tr>
</table>


#16. Copy data frames to Spark from H2O
    df_weather = hc.as_spark_frame(f_weather,)
    df_census = hc.as_spark_frame(f_census)
    df_crimes = hc.as_spark_frame(f_crimes)


#17. Look at the weather data as parsed in Spark 
(only showing top 2 rows)

    df_weather.show(2)

    +-----+---+----+-------+--------+-------+
    |month|day|year|maxTemp|meanTemp|minTemp|
    +-----+---+----+-------+--------+-------+
    |    1|  1|2001|     23|      14|      6|
    |    1|  2|2001|     18|      12|      6|
    +-----+---+----+-------+--------+-------+

#18. Join columns from Crime, Census and Weather DataFrames in Spark
    
    ## Register DataFrames as tables in SQL context
    sqlContext.registerDataFrameAsTable(df_weather, "chicagoWeather")
    sqlContext.registerDataFrameAsTable(df_census, "chicagoCensus")
    sqlContext.registerDataFrameAsTable(df_crimes, "chicagoCrime")
    
    
    crimeWithWeather = sqlContext.sql("""SELECT
    a.Year, a.Month, a.Day, a.WeekNum, a.HourOfDay, a.Weekend, a.Season, a.WeekDay,
    a.IUCR, a.Primary_Type, a.Location_Description, a.Community_Area, a.District,
    a.Arrest, a.Domestic, a.Beat, a.Ward, a.FBI_Code,
    b.minTemp, b.maxTemp, b.meanTemp,
    c.PERCENT_AGED_UNDER_18_OR_OVER_64, c.PER_CAPITA_INCOME, c.HARDSHIP_INDEX,
    c.PERCENT_OF_HOUSING_CROWDED, c.PERCENT_HOUSEHOLDS_BELOW_POVERTY,
    c.PERCENT_AGED_16__UNEMPLOYED, c.PERCENT_AGED_25__WITHOUT_HIGH_SCHOOL_DIPLOMA
    FROM chicagoCrime a
    JOIN chicagoWeather b
    ON a.Year = b.year AND a.Month = b.month AND a.Day = b.day
    JOIN chicagoCensus c
    ON a.Community_Area = c.Community_Area_Number""")


#19. Print the `crimeWithWeather` data table from Spark
    crimeWithWeather.show(2)

    +----+-----+---+-------+---------+-------+------+-------+----+-----------------+--------------------+--------------+--------+------+--------+----+----+--------+-------+-------+--------+--------------------------------+-----------------+--------------+--------------------------+--------------------------------+---------------------------+--------------------------------------------+
    |Year|Month|Day|WeekNum|HourOfDay|Weekend|Season|WeekDay|IUCR|     Primary_Type|Location_Description|Community_Area|District|Arrest|Domestic|Beat|Ward|FBI_Code|minTemp|maxTemp|meanTemp|PERCENT_AGED_UNDER_18_OR_OVER_64|PER_CAPITA_INCOME|HARDSHIP_INDEX|PERCENT_OF_HOUSING_CROWDED|PERCENT_HOUSEHOLDS_BELOW_POVERTY|PERCENT_AGED_16__UNEMPLOYED|PERCENT_AGED_25__WITHOUT_HIGH_SCHOOL_DIPLOMA|
    +----+-----+---+-------+---------+-------+------+-------+----+-----------------+--------------------+--------------+--------+------+--------+----+----+--------+-------+-------+--------+--------------------------------+-----------------+--------------+--------------------------+--------------------------------+---------------------------+--------------------------------------------+
    |2015|    1| 23|      4|       22|      0|Winter|    Fri|143A|WEAPONS VIOLATION|               ALLEY|            31|      12|  true|   false|1234|  25|      15|     29|     31|      30|                            32.6|            16444|            76|         9.600000000000001|                            25.8|                       15.8|                                        40.7|
    |2015|    1| 23|      4|       19|      0|Winter|    Fri|4625|    OTHER OFFENSE|            SIDEWALK|            31|      10|  true|   false|1034|  25|      26|     29|     31|      30|                            32.6|            16444|            76|         9.600000000000001|                            25.8|                       15.8|                                        40.7|
    +----+-----+---+-------+---------+-------+------+-------+----+-----------------+--------------------+--------------+--------+------+--------+----+----+--------+-------+-------+--------+--------------------------------+-----------------+--------------+--------------------------+--------------------------------+---------------------------+--------------------------------------------+
    only showing top 2 rows
    


#20. Copy table from Spark to H2O
    hc.as_h2o_frame?
    crimeWithWeatherHF = hc.as_h2o_frame(crimeWithWeather,framename="crimeWithWeather")

    H2OContext: ip=172.16.2.98, port=54329



    crimeWithWeatherHF.summary()


<table>
<tr><th>       </th><th>Year  </th><th>Month         </th><th>Day          </th><th>WeekNum       </th><th>HourOfDay    </th><th>Weekend       </th><th>Season  </th><th>WeekDay  </th><th>IUCR  </th><th>Primary_Type              </th><th>Location_Description          </th><th>Community_Area  </th><th>District     </th><th>Arrest  </th><th>Domestic  </th><th>Beat         </th><th>Ward         </th><th>FBI_Code  </th><th>minTemp      </th><th>maxTemp      </th><th>meanTemp     </th><th>PERCENT_AGED_UNDER_18_OR_OVER_64  </th><th>PER_CAPITA_INCOME  </th><th>HARDSHIP_INDEX  </th><th>PERCENT_OF_HOUSING_CROWDED  </th><th>PERCENT_HOUSEHOLDS_BELOW_POVERTY  </th><th>PERCENT_AGED_16__UNEMPLOYED  </th><th>PERCENT_AGED_25__WITHOUT_HIGH_SCHOOL_DIPLOMA  </th></tr>
<tr><td>type   </td><td>int   </td><td>int           </td><td>int          </td><td>int           </td><td>int          </td><td>int           </td><td>string  </td><td>string   </td><td>string</td><td>string                    </td><td>string                        </td><td>int             </td><td>int          </td><td>string  </td><td>string    </td><td>int          </td><td>int          </td><td>string    </td><td>int          </td><td>int          </td><td>int          </td><td>real                              </td><td>int                </td><td>int             </td><td>real                        </td><td>real                              </td><td>real                         </td><td>real                                          </td></tr>
<tr><td>mins   </td><td>2015.0</td><td>1.0           </td><td>1.0          </td><td>4.0           </td><td>0.0          </td><td>0.0           </td><td>NaN     </td><td>NaN      </td><td>NaN   </td><td>NaN                       </td><td>NaN                           </td><td>1.0             </td><td>1.0          </td><td>NaN     </td><td>NaN       </td><td>111.0        </td><td>1.0          </td><td>NaN       </td><td>-2.0         </td><td>15.0         </td><td>7.0          </td><td>13.5                              </td><td>8201.0             </td><td>1.0             </td><td>0.3                         </td><td>3.3                               </td><td>4.7                          </td><td>2.5                                           </td></tr>
<tr><td>mean   </td><td>2015.0</td><td>1.41944194419 </td><td>17.6839683968</td><td>5.18081808181 </td><td>13.6319631963</td><td>0.159115911591</td><td>NaN     </td><td>NaN      </td><td>NaN   </td><td>NaN                       </td><td>NaN                           </td><td>37.4476447645   </td><td>11.3489885128</td><td>NaN     </td><td>NaN       </td><td>1159.61806181</td><td>22.9540954095</td><td>NaN       </td><td>17.699669967 </td><td>31.7199719972</td><td>24.9408940894</td><td>35.0596759676                     </td><td>25221.3057306      </td><td>54.4786478648   </td><td>5.43707370737               </td><td>24.600750075                      </td><td>16.8288328833                </td><td>21.096639664                                  </td></tr>
<tr><td>maxs   </td><td>2015.0</td><td>2.0           </td><td>31.0         </td><td>6.0           </td><td>23.0         </td><td>1.0           </td><td>NaN     </td><td>NaN      </td><td>NaN   </td><td>NaN                       </td><td>NaN                           </td><td>77.0            </td><td>25.0         </td><td>NaN     </td><td>NaN       </td><td>2535.0       </td><td>50.0         </td><td>NaN       </td><td>29.0         </td><td>43.0         </td><td>36.0         </td><td>51.5                              </td><td>88669.0            </td><td>98.0            </td><td>15.8                        </td><td>56.5                              </td><td>35.9                         </td><td>54.8                                          </td></tr>
<tr><td>sigma  </td><td>0.0   </td><td>0.493492406787</td><td>11.1801043358</td><td>0.738929830409</td><td>6.47321735807</td><td>0.365802434041</td><td>NaN     </td><td>NaN      </td><td>NaN   </td><td>NaN                       </td><td>NaN                           </td><td>21.2748762223   </td><td>6.94547493301</td><td>NaN     </td><td>NaN       </td><td>695.76029875 </td><td>13.6495661144</td><td>NaN       </td><td>8.96118136438</td><td>6.93809913472</td><td>7.46302527062</td><td>7.95653388237                     </td><td>18010.0446225      </td><td>29.3247456472   </td><td>3.75289588494               </td><td>10.1450570661                     </td><td>7.58926327988                </td><td>11.3868817911                                 </td></tr>
<tr><td>zeros  </td><td>0     </td><td>0             </td><td>0            </td><td>0             </td><td>374          </td><td>8408          </td><td>0       </td><td>0        </td><td>0     </td><td>0                         </td><td>0                             </td><td>0               </td><td>0            </td><td>0       </td><td>0         </td><td>0            </td><td>0            </td><td>0         </td><td>0            </td><td>0            </td><td>0            </td><td>0                                 </td><td>0                  </td><td>0               </td><td>0                           </td><td>0                                 </td><td>0                            </td><td>0                                             </td></tr>
<tr><td>missing</td><td>0     </td><td>0             </td><td>0            </td><td>0             </td><td>0            </td><td>0             </td><td>0       </td><td>0        </td><td>0     </td><td>0                         </td><td>6                             </td><td>0               </td><td>162          </td><td>0       </td><td>0         </td><td>0            </td><td>0            </td><td>0         </td><td>0            </td><td>0            </td><td>0            </td><td>0                                 </td><td>0                  </td><td>0               </td><td>0                           </td><td>0                                 </td><td>0                            </td><td>0                                             </td></tr>
<tr><td>0      </td><td>2015.0</td><td>1.0           </td><td>24.0         </td><td>4.0           </td><td>22.0         </td><td>0.0           </td><td>Winter  </td><td>Sat      </td><td>2820  </td><td>OTHER OFFENSE             </td><td>APARTMENT                     </td><td>31.0            </td><td>10.0         </td><td>false   </td><td>false     </td><td>1034.0       </td><td>25.0         </td><td>26        </td><td>29.0         </td><td>43.0         </td><td>36.0         </td><td>32.6                              </td><td>16444.0            </td><td>76.0            </td><td>9.6                         </td><td>25.8                              </td><td>15.8                         </td><td>40.7                                          </td></tr>
<tr><td>1      </td><td>2015.0</td><td>1.0           </td><td>24.0         </td><td>4.0           </td><td>21.0         </td><td>0.0           </td><td>Winter  </td><td>Sat      </td><td>1310  </td><td>CRIMINAL DAMAGE           </td><td>RESTAURANT                    </td><td>31.0            </td><td>12.0         </td><td>true    </td><td>false     </td><td>1233.0       </td><td>25.0         </td><td>14        </td><td>29.0         </td><td>43.0         </td><td>36.0         </td><td>32.6                              </td><td>16444.0            </td><td>76.0            </td><td>9.6                         </td><td>25.8                              </td><td>15.8                         </td><td>40.7                                          </td></tr>
<tr><td>2      </td><td>2015.0</td><td>1.0           </td><td>24.0         </td><td>4.0           </td><td>18.0         </td><td>0.0           </td><td>Winter  </td><td>Sat      </td><td>1750  </td><td>OFFENSE INVOLVING CHILDREN</td><td>RESIDENCE                     </td><td>31.0            </td><td>12.0         </td><td>false   </td><td>true      </td><td>1235.0       </td><td>25.0         </td><td>20        </td><td>29.0         </td><td>43.0         </td><td>36.0         </td><td>32.6                              </td><td>16444.0            </td><td>76.0            </td><td>9.6                         </td><td>25.8                              </td><td>15.8                         </td><td>40.7                                          </td></tr>
<tr><td>3      </td><td>2015.0</td><td>1.0           </td><td>24.0         </td><td>4.0           </td><td>18.0         </td><td>0.0           </td><td>Winter  </td><td>Sat      </td><td>0460  </td><td>BATTERY                   </td><td>OTHER                         </td><td>31.0            </td><td>10.0         </td><td>false   </td><td>false     </td><td>1023.0       </td><td>25.0         </td><td>08B       </td><td>29.0         </td><td>43.0         </td><td>36.0         </td><td>32.6                              </td><td>16444.0            </td><td>76.0            </td><td>9.6                         </td><td>25.8                              </td><td>15.8                         </td><td>40.7                                          </td></tr>
<tr><td>4      </td><td>2015.0</td><td>1.0           </td><td>24.0         </td><td>4.0           </td><td>13.0         </td><td>0.0           </td><td>Winter  </td><td>Sat      </td><td>0890  </td><td>THEFT                     </td><td>CURRENCY EXCHANGE             </td><td>31.0            </td><td>10.0         </td><td>false   </td><td>false     </td><td>1023.0       </td><td>25.0         </td><td>06        </td><td>29.0         </td><td>43.0         </td><td>36.0         </td><td>32.6                              </td><td>16444.0            </td><td>76.0            </td><td>9.6                         </td><td>25.8                              </td><td>15.8                         </td><td>40.7                                          </td></tr>
<tr><td>5      </td><td>2015.0</td><td>1.0           </td><td>24.0         </td><td>4.0           </td><td>9.0          </td><td>0.0           </td><td>Winter  </td><td>Sat      </td><td>0560  </td><td>ASSAULT                   </td><td>OTHER                         </td><td>31.0            </td><td>12.0         </td><td>false   </td><td>false     </td><td>1234.0       </td><td>25.0         </td><td>08A       </td><td>29.0         </td><td>43.0         </td><td>36.0         </td><td>32.6                              </td><td>16444.0            </td><td>76.0            </td><td>9.6                         </td><td>25.8                              </td><td>15.8                         </td><td>40.7                                          </td></tr>
<tr><td>6      </td><td>2015.0</td><td>1.0           </td><td>24.0         </td><td>4.0           </td><td>8.0          </td><td>0.0           </td><td>Winter  </td><td>Sat      </td><td>0486  </td><td>BATTERY                   </td><td>RESIDENCE                     </td><td>31.0            </td><td>12.0         </td><td>true    </td><td>true      </td><td>1235.0       </td><td>25.0         </td><td>08B       </td><td>29.0         </td><td>43.0         </td><td>36.0         </td><td>32.6                              </td><td>16444.0            </td><td>76.0            </td><td>9.6                         </td><td>25.8                              </td><td>15.8                         </td><td>40.7                                          </td></tr>
<tr><td>7      </td><td>2015.0</td><td>1.0           </td><td>24.0         </td><td>4.0           </td><td>1.0          </td><td>0.0           </td><td>Winter  </td><td>Sat      </td><td>0420  </td><td>BATTERY                   </td><td>SIDEWALK                      </td><td>31.0            </td><td>10.0         </td><td>false   </td><td>false     </td><td>1034.0       </td><td>25.0         </td><td>04B       </td><td>29.0         </td><td>43.0         </td><td>36.0         </td><td>32.6                              </td><td>16444.0            </td><td>76.0            </td><td>9.6                         </td><td>25.8                              </td><td>15.8                         </td><td>40.7                                          </td></tr>
<tr><td>8      </td><td>2015.0</td><td>1.0           </td><td>24.0         </td><td>4.0           </td><td>0.0          </td><td>0.0           </td><td>Winter  </td><td>Sat      </td><td>1320  </td><td>CRIMINAL DAMAGE           </td><td>PARKING LOT/GARAGE(NON.RESID.)</td><td>31.0            </td><td>9.0          </td><td>false   </td><td>false     </td><td>912.0        </td><td>11.0         </td><td>14        </td><td>29.0         </td><td>43.0         </td><td>36.0         </td><td>32.6                              </td><td>16444.0            </td><td>76.0            </td><td>9.6                         </td><td>25.8                              </td><td>15.8                         </td><td>40.7                                          </td></tr>
<tr><td>9      </td><td>2015.0</td><td>1.0           </td><td>31.0         </td><td>5.0           </td><td>23.0         </td><td>0.0           </td><td>Winter  </td><td>Sat      </td><td>0820  </td><td>THEFT                     </td><td>SIDEWALK                      </td><td>31.0            </td><td>12.0         </td><td>false   </td><td>false     </td><td>1234.0       </td><td>25.0         </td><td>06        </td><td>19.0         </td><td>36.0         </td><td>28.0         </td><td>32.6                              </td><td>16444.0            </td><td>76.0            </td><td>9.6                         </td><td>25.8                              </td><td>15.8                         </td><td>40.7                                          </td></tr>
</table>



#21. Assign column types to the `CrimeWeatherHF` data table in H2O
    crimeWithWeatherHF["Season"]= crimeWithWeatherHF["Season"].asfactor()
    crimeWithWeatherHF["WeekDay"]= crimeWithWeatherHF["WeekDay"].asfactor()
    crimeWithWeatherHF["IUCR"]= crimeWithWeatherHF["IUCR"].asfactor()
    crimeWithWeatherHF["Primary_Type"]= crimeWithWeatherHF["Primary_Type"].asfactor()
    crimeWithWeatherHF["Location_Description"]= crimeWithWeatherHF["Location_Description"].asfactor()
    crimeWithWeatherHF["Arrest"]= crimeWithWeatherHF["Arrest"].asfactor()
    crimeWithWeatherHF["Domestic"]= crimeWithWeatherHF["Domestic"].asfactor()
    crimeWithWeatherHF["FBI_Code"]= crimeWithWeatherHF["FBI_Code"].asfactor()
    crimeWithWeatherHF["Season"]= crimeWithWeatherHF["Season"].asfactor()



    crimeWithWeatherHF.summary()


<table>
<tr><th>       </th><th>Year  </th><th>Month         </th><th>Day          </th><th>WeekNum       </th><th>HourOfDay    </th><th>Weekend       </th><th>Season  </th><th>WeekDay  </th><th>IUCR  </th><th>Primary_Type              </th><th>Location_Description          </th><th>Community_Area  </th><th>District     </th><th>Arrest        </th><th>Domestic      </th><th>Beat         </th><th>Ward         </th><th>FBI_Code  </th><th>minTemp      </th><th>maxTemp      </th><th>meanTemp     </th><th>PERCENT_AGED_UNDER_18_OR_OVER_64  </th><th>PER_CAPITA_INCOME  </th><th>HARDSHIP_INDEX  </th><th>PERCENT_OF_HOUSING_CROWDED  </th><th>PERCENT_HOUSEHOLDS_BELOW_POVERTY  </th><th>PERCENT_AGED_16__UNEMPLOYED  </th><th>PERCENT_AGED_25__WITHOUT_HIGH_SCHOOL_DIPLOMA  </th></tr>
<tr><td>type   </td><td>int   </td><td>int           </td><td>int          </td><td>int           </td><td>int          </td><td>int           </td><td>enum    </td><td>enum     </td><td>enum  </td><td>enum                      </td><td>enum                          </td><td>int             </td><td>int          </td><td>enum          </td><td>enum          </td><td>int          </td><td>int          </td><td>enum      </td><td>int          </td><td>int          </td><td>int          </td><td>real                              </td><td>int                </td><td>int             </td><td>real                        </td><td>real                              </td><td>real                         </td><td>real                                          </td></tr>
<tr><td>mins   </td><td>2015.0</td><td>1.0           </td><td>1.0          </td><td>4.0           </td><td>0.0          </td><td>0.0           </td><td>0.0     </td><td>0.0      </td><td>0.0   </td><td>0.0                       </td><td>0.0                           </td><td>1.0             </td><td>1.0          </td><td>0.0           </td><td>0.0           </td><td>111.0        </td><td>1.0          </td><td>0.0       </td><td>-2.0         </td><td>15.0         </td><td>7.0          </td><td>13.5                              </td><td>8201.0             </td><td>1.0             </td><td>0.3                         </td><td>3.3                               </td><td>4.7                          </td><td>2.5                                           </td></tr>
<tr><td>mean   </td><td>2015.0</td><td>1.41944194419 </td><td>17.6839683968</td><td>5.18081808181 </td><td>13.6319631963</td><td>0.159115911591</td><td>0.0     </td><td>NaN      </td><td>NaN   </td><td>NaN                       </td><td>NaN                           </td><td>37.4476447645   </td><td>11.3489885128</td><td>0.292829282928</td><td>0.152315231523</td><td>1159.61806181</td><td>22.9540954095</td><td>NaN       </td><td>17.699669967 </td><td>31.7199719972</td><td>24.9408940894</td><td>35.0596759676                     </td><td>25221.3057306      </td><td>54.4786478648   </td><td>5.43707370737               </td><td>24.600750075                      </td><td>16.8288328833                </td><td>21.096639664                                  </td></tr>
<tr><td>maxs   </td><td>2015.0</td><td>2.0           </td><td>31.0         </td><td>6.0           </td><td>23.0         </td><td>1.0           </td><td>0.0     </td><td>6.0      </td><td>212.0 </td><td>26.0                      </td><td>90.0                          </td><td>77.0            </td><td>25.0         </td><td>1.0           </td><td>1.0           </td><td>2535.0       </td><td>50.0         </td><td>24.0      </td><td>29.0         </td><td>43.0         </td><td>36.0         </td><td>51.5                              </td><td>88669.0            </td><td>98.0            </td><td>15.8                        </td><td>56.5                              </td><td>35.9                         </td><td>54.8                                          </td></tr>
<tr><td>sigma  </td><td>0.0   </td><td>0.493492406787</td><td>11.1801043358</td><td>0.738929830409</td><td>6.47321735807</td><td>0.365802434041</td><td>0.0     </td><td>NaN      </td><td>NaN   </td><td>NaN                       </td><td>NaN                           </td><td>21.2748762223   </td><td>6.94547493301</td><td>0.455083515588</td><td>0.35934414686 </td><td>695.76029875 </td><td>13.6495661144</td><td>NaN       </td><td>8.96118136438</td><td>6.93809913472</td><td>7.46302527062</td><td>7.95653388237                     </td><td>18010.0446225      </td><td>29.3247456472   </td><td>3.75289588494               </td><td>10.1450570661                     </td><td>7.58926327988                </td><td>11.3868817911                                 </td></tr>
<tr><td>zeros  </td><td>0     </td><td>0             </td><td>0            </td><td>0             </td><td>374          </td><td>8408          </td><td>9999    </td><td>1942     </td><td>16    </td><td>11                        </td><td>19                            </td><td>0               </td><td>0            </td><td>7071          </td><td>8476          </td><td>0            </td><td>0            </td><td>16        </td><td>0            </td><td>0            </td><td>0            </td><td>0                                 </td><td>0                  </td><td>0               </td><td>0                           </td><td>0                                 </td><td>0                            </td><td>0                                             </td></tr>
<tr><td>missing</td><td>0     </td><td>0             </td><td>0            </td><td>0             </td><td>0            </td><td>0             </td><td>0       </td><td>0        </td><td>0     </td><td>0                         </td><td>6                             </td><td>0               </td><td>162          </td><td>0             </td><td>0             </td><td>0            </td><td>0            </td><td>0         </td><td>0            </td><td>0            </td><td>0            </td><td>0                                 </td><td>0                  </td><td>0               </td><td>0                           </td><td>0                                 </td><td>0                            </td><td>0                                             </td></tr>
<tr><td>0      </td><td>2015.0</td><td>1.0           </td><td>24.0         </td><td>4.0           </td><td>22.0         </td><td>0.0           </td><td>Winter  </td><td>Sat      </td><td>2820  </td><td>OTHER OFFENSE             </td><td>APARTMENT                     </td><td>31.0            </td><td>10.0         </td><td>false         </td><td>false         </td><td>1034.0       </td><td>25.0         </td><td>26        </td><td>29.0         </td><td>43.0         </td><td>36.0         </td><td>32.6                              </td><td>16444.0            </td><td>76.0            </td><td>9.6                         </td><td>25.8                              </td><td>15.8                         </td><td>40.7                                          </td></tr>
<tr><td>1      </td><td>2015.0</td><td>1.0           </td><td>24.0         </td><td>4.0           </td><td>21.0         </td><td>0.0           </td><td>Winter  </td><td>Sat      </td><td>1310  </td><td>CRIMINAL DAMAGE           </td><td>RESTAURANT                    </td><td>31.0            </td><td>12.0         </td><td>true          </td><td>false         </td><td>1233.0       </td><td>25.0         </td><td>14        </td><td>29.0         </td><td>43.0         </td><td>36.0         </td><td>32.6                              </td><td>16444.0            </td><td>76.0            </td><td>9.6                         </td><td>25.8                              </td><td>15.8                         </td><td>40.7                                          </td></tr>
<tr><td>2      </td><td>2015.0</td><td>1.0           </td><td>24.0         </td><td>4.0           </td><td>18.0         </td><td>0.0           </td><td>Winter  </td><td>Sat      </td><td>1750  </td><td>OFFENSE INVOLVING CHILDREN</td><td>RESIDENCE                     </td><td>31.0            </td><td>12.0         </td><td>false         </td><td>true          </td><td>1235.0       </td><td>25.0         </td><td>20        </td><td>29.0         </td><td>43.0         </td><td>36.0         </td><td>32.6                              </td><td>16444.0            </td><td>76.0            </td><td>9.6                         </td><td>25.8                              </td><td>15.8                         </td><td>40.7                                          </td></tr>
<tr><td>3      </td><td>2015.0</td><td>1.0           </td><td>24.0         </td><td>4.0           </td><td>18.0         </td><td>0.0           </td><td>Winter  </td><td>Sat      </td><td>0460  </td><td>BATTERY                   </td><td>OTHER                         </td><td>31.0            </td><td>10.0         </td><td>false         </td><td>false         </td><td>1023.0       </td><td>25.0         </td><td>08B       </td><td>29.0         </td><td>43.0         </td><td>36.0         </td><td>32.6                              </td><td>16444.0            </td><td>76.0            </td><td>9.6                         </td><td>25.8                              </td><td>15.8                         </td><td>40.7                                          </td></tr>
<tr><td>4      </td><td>2015.0</td><td>1.0           </td><td>24.0         </td><td>4.0           </td><td>13.0         </td><td>0.0           </td><td>Winter  </td><td>Sat      </td><td>0890  </td><td>THEFT                     </td><td>CURRENCY EXCHANGE             </td><td>31.0            </td><td>10.0         </td><td>false         </td><td>false         </td><td>1023.0       </td><td>25.0         </td><td>06        </td><td>29.0         </td><td>43.0         </td><td>36.0         </td><td>32.6                              </td><td>16444.0            </td><td>76.0            </td><td>9.6                         </td><td>25.8                              </td><td>15.8                         </td><td>40.7                                          </td></tr>
<tr><td>5      </td><td>2015.0</td><td>1.0           </td><td>24.0         </td><td>4.0           </td><td>9.0          </td><td>0.0           </td><td>Winter  </td><td>Sat      </td><td>0560  </td><td>ASSAULT                   </td><td>OTHER                         </td><td>31.0            </td><td>12.0         </td><td>false         </td><td>false         </td><td>1234.0       </td><td>25.0         </td><td>08A       </td><td>29.0         </td><td>43.0         </td><td>36.0         </td><td>32.6                              </td><td>16444.0            </td><td>76.0            </td><td>9.6                         </td><td>25.8                              </td><td>15.8                         </td><td>40.7                                          </td></tr>
<tr><td>6      </td><td>2015.0</td><td>1.0           </td><td>24.0         </td><td>4.0           </td><td>8.0          </td><td>0.0           </td><td>Winter  </td><td>Sat      </td><td>0486  </td><td>BATTERY                   </td><td>RESIDENCE                     </td><td>31.0            </td><td>12.0         </td><td>true          </td><td>true          </td><td>1235.0       </td><td>25.0         </td><td>08B       </td><td>29.0         </td><td>43.0         </td><td>36.0         </td><td>32.6                              </td><td>16444.0            </td><td>76.0            </td><td>9.6                         </td><td>25.8                              </td><td>15.8                         </td><td>40.7                                          </td></tr>
<tr><td>7      </td><td>2015.0</td><td>1.0           </td><td>24.0         </td><td>4.0           </td><td>1.0          </td><td>0.0           </td><td>Winter  </td><td>Sat      </td><td>0420  </td><td>BATTERY                   </td><td>SIDEWALK                      </td><td>31.0            </td><td>10.0         </td><td>false         </td><td>false         </td><td>1034.0       </td><td>25.0         </td><td>04B       </td><td>29.0         </td><td>43.0         </td><td>36.0         </td><td>32.6                              </td><td>16444.0            </td><td>76.0            </td><td>9.6                         </td><td>25.8                              </td><td>15.8                         </td><td>40.7                                          </td></tr>
<tr><td>8      </td><td>2015.0</td><td>1.0           </td><td>24.0         </td><td>4.0           </td><td>0.0          </td><td>0.0           </td><td>Winter  </td><td>Sat      </td><td>1320  </td><td>CRIMINAL DAMAGE           </td><td>PARKING LOT/GARAGE(NON.RESID.)</td><td>31.0            </td><td>9.0          </td><td>false         </td><td>false         </td><td>912.0        </td><td>11.0         </td><td>14        </td><td>29.0         </td><td>43.0         </td><td>36.0         </td><td>32.6                              </td><td>16444.0            </td><td>76.0            </td><td>9.6                         </td><td>25.8                              </td><td>15.8                         </td><td>40.7                                          </td></tr>
<tr><td>9      </td><td>2015.0</td><td>1.0           </td><td>31.0         </td><td>5.0           </td><td>23.0         </td><td>0.0           </td><td>Winter  </td><td>Sat      </td><td>0820  </td><td>THEFT                     </td><td>SIDEWALK                      </td><td>31.0            </td><td>12.0         </td><td>false         </td><td>false         </td><td>1234.0       </td><td>25.0         </td><td>06        </td><td>19.0         </td><td>36.0         </td><td>28.0         </td><td>32.6                              </td><td>16444.0            </td><td>76.0            </td><td>9.6                         </td><td>25.8                              </td><td>15.8                         </td><td>40.7                                          </td></tr>
</table>



#22. Split final H2O data table into train test and validation sets
    ratios = [0.6,0.2]
    frs = crimeWithWeatherHF.split_frame(ratios,seed=12345)
    train = frs[0]
    train.frame_id = "Train"
    valid = frs[2]
    valid.frame_id = "Validation"
    test = frs[1]
    test.frame_id = "Test"


#23. Import Model Builders from H2O Python
    from h2o.estimators.gbm import H2OGradientBoostingEstimator
    from h2o.estimators.deeplearning import H2ODeepLearningEstimator


#24. Inspect the availble GBM parameters
    H2OGradientBoostingEstimator?


#25. Define Predictors
    predictors = crimeWithWeatherHF.names[:]
    response = "Arrest"
    predictors.remove(response)


#26. Create a Simple GBM model to Predict Arrests
    model_gbm = H2OGradientBoostingEstimator(ntrees         =50,
                                            max_depth      =6,
                                            learn_rate     =0.1, 
                                            #nfolds         =2,
                                            distribution   ="bernoulli")
    
    model_gbm.train(x               =predictors,
                   y               ="Arrest",
                   training_frame  =train,
                   validation_frame=valid
                   )
    
#27. Create a Simple Deep Learning model to Predict Arrests
    model_dl = H2ODeepLearningEstimator(variable_importances=True,
                                       loss                ="Automatic")
    
    model_dl.train(x                =predictors,
                  y                ="Arrest",
                  training_frame  =train,
                  validation_frame=valid)

    
    gbm Model Build Progress: [##################################################] 100%
    
    deeplearning Model Build Progress: [##################################################] 100%



#28. Print confusion matrices for the training and validation datasets
    print(model_gbm.confusion_matrix(train = True))
    print(model_gbm.confusion_matrix(valid = True))

    
**Confusion Matrix (Act/Pred) for max f1 @ threshold = 0.335827722991:**



<div style="overflow:auto"><table style="width:50%"><tr><td><b></b></td>
<td><b>false</b></td>
<td><b>true</b></td>
<td><b>Error</b></td>
<td><b>Rate</b></td></tr>
<tr><td>false</td>
<td>4125.0</td>
<td>142.0</td>
<td>0.0333</td>
<td> (142.0/4267.0)</td></tr>
<tr><td>true</td>
<td>251.0</td>
<td>1504.0</td>
<td>0.143</td>
<td> (251.0/1755.0)</td></tr>
<tr><td>Total</td>
<td>4376.0</td>
<td>1646.0</td>
<td>0.0653</td>
<td> (393.0/6022.0)</td></tr></table></div>


    
    
**Confusion Matrix (Act/Pred) for max f1 @ threshold = 0.432844055866:**



<div style="overflow:auto"><table style="width:50%"><tr><td><b></b></td>
<td><b>false</b></td>
<td><b>true</b></td>
<td><b>Error</b></td>
<td><b>Rate</b></td></tr>
<tr><td>false</td>
<td>1362.0</td>
<td>61.0</td>
<td>0.0429</td>
<td> (61.0/1423.0)</td></tr>
<tr><td>true</td>
<td>150.0</td>
<td>443.0</td>
<td>0.253</td>
<td> (150.0/593.0)</td></tr>
<tr><td>Total</td>
<td>1512.0</td>
<td>504.0</td>
<td>0.1047</td>
<td> (211.0/2016.0)</td></tr></table></div>


    



    print(model_gbm.auc(train=True))
    print(model_gbm.auc(valid=True))
    model_gbm.plot(metric="AUC")

    0.974667176776
    0.92596751276



#29. Print variable importances
    model_gbm.varimp(True)




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variable</th>
      <th>relative_importance</th>
      <th>scaled_importance</th>
      <th>percentage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>IUCR</td>
      <td>4280.939453</td>
      <td>1.000000e+00</td>
      <td>8.234218e-01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Location_Description</td>
      <td>487.323059</td>
      <td>1.138355e-01</td>
      <td>9.373466e-02</td>
    </tr>
    <tr>
      <th>2</th>
      <td>WeekDay</td>
      <td>55.790558</td>
      <td>1.303232e-02</td>
      <td>1.073109e-02</td>
    </tr>
    <tr>
      <th>3</th>
      <td>HourOfDay</td>
      <td>55.419220</td>
      <td>1.294557e-02</td>
      <td>1.065967e-02</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PERCENT_AGED_16__UNEMPLOYED</td>
      <td>34.422894</td>
      <td>8.040967e-03</td>
      <td>6.621107e-03</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Beat</td>
      <td>31.468222</td>
      <td>7.350775e-03</td>
      <td>6.052788e-03</td>
    </tr>
    <tr>
      <th>6</th>
      <td>PERCENT_HOUSEHOLDS_BELOW_POVERTY</td>
      <td>29.103352</td>
      <td>6.798356e-03</td>
      <td>5.597915e-03</td>
    </tr>
    <tr>
      <th>7</th>
      <td>PER_CAPITA_INCOME</td>
      <td>26.233143</td>
      <td>6.127894e-03</td>
      <td>5.045841e-03</td>
    </tr>
    <tr>
      <th>8</th>
      <td>PERCENT_AGED_UNDER_18_OR_OVER_64</td>
      <td>24.077402</td>
      <td>5.624327e-03</td>
      <td>4.631193e-03</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Day</td>
      <td>23.472567</td>
      <td>5.483041e-03</td>
      <td>4.514855e-03</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>15</th>
      <td>maxTemp</td>
      <td>11.300793</td>
      <td>2.639793e-03</td>
      <td>2.173663e-03</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Community_Area</td>
      <td>10.252146</td>
      <td>2.394835e-03</td>
      <td>1.971960e-03</td>
    </tr>
    <tr>
      <th>17</th>
      <td>HARDSHIP_INDEX</td>
      <td>10.116072</td>
      <td>2.363049e-03</td>
      <td>1.945786e-03</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Domestic</td>
      <td>9.294327</td>
      <td>2.171095e-03</td>
      <td>1.787727e-03</td>
    </tr>
    <tr>
      <th>19</th>
      <td>District</td>
      <td>8.304654</td>
      <td>1.939914e-03</td>
      <td>1.597367e-03</td>
    </tr>
    <tr>
      <th>20</th>
      <td>minTemp</td>
      <td>6.243027</td>
      <td>1.458331e-03</td>
      <td>1.200822e-03</td>
    </tr>
    <tr>
      <th>21</th>
      <td>WeekNum</td>
      <td>4.230102</td>
      <td>9.881246e-04</td>
      <td>8.136433e-04</td>
    </tr>
    <tr>
      <th>22</th>
      <td>FBI_Code</td>
      <td>2.363182</td>
      <td>5.520241e-04</td>
      <td>4.545486e-04</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Month</td>
      <td>0.000018</td>
      <td>4.187325e-09</td>
      <td>3.447935e-09</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Weekend</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
  </tbody>
</table>
<p>25 rows × 4 columns</p>
</div>



#30. Inspect Deep Learning model output
    model_dl

    Model Details
    =============
    H2ODeepLearningEstimator :  Deep Learning
    Model Key:  DeepLearning_model_python_1446861372065_4
    
    Status of Neuron Layers: predicting Arrest, 2-class classification, bernoulli distribution, CrossEntropy loss, 118,802 weights/biases, 1.4 MB, 72,478 training samples, mini-batch size 1
    



<div style="overflow:auto"><table style="width:50%"><tr><td><b></b></td>
<td><b>layer</b></td>
<td><b>units</b></td>
<td><b>type</b></td>
<td><b>dropout</b></td>
<td><b>l1</b></td>
<td><b>l2</b></td>
<td><b>mean_rate</b></td>
<td><b>rate_RMS</b></td>
<td><b>momentum</b></td>
<td><b>mean_weight</b></td>
<td><b>weight_RMS</b></td>
<td><b>mean_bias</b></td>
<td><b>bias_RMS</b></td></tr>
<tr><td></td>
<td>1</td>
<td>390</td>
<td>Input</td>
<td>0.0</td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td>
<td></td></tr>
<tr><td></td>
<td>2</td>
<td>200</td>
<td>Rectifier</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.1</td>
<td>0.3</td>
<td>0.0</td>
<td>-0.0</td>
<td>0.1</td>
<td>-0.0</td>
<td>0.1</td></tr>
<tr><td></td>
<td>3</td>
<td>200</td>
<td>Rectifier</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.1</td>
<td>0.2</td>
<td>0.0</td>
<td>-0.0</td>
<td>0.1</td>
<td>0.8</td>
<td>0.2</td></tr>
<tr><td></td>
<td>4</td>
<td>2</td>
<td>Softmax</td>
<td></td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td>
<td>0.4</td>
<td>-0.0</td>
<td>0.0</td></tr></table></div>


    
    
**ModelMetricsBinomial: deeplearning**

  Reported on train data. 
    
    MSE: 0.0737426129728
    R^2: 0.642891439669
    LogLoss: 0.242051500943
    AUC: 0.950131166302
    Gini: 0.900262332604
    
**Confusion Matrix (Act/Pred) for max f1 @ threshold = 0.343997370612:**



<div style="overflow:auto"><table style="width:50%"><tr><td><b></b></td>
<td><b>false</b></td>
<td><b>true</b></td>
<td><b>Error</b></td>
<td><b>Rate</b></td></tr>
<tr><td>false</td>
<td>4003.0</td>
<td>264.0</td>
<td>0.0619</td>
<td> (264.0/4267.0)</td></tr>
<tr><td>true</td>
<td>358.0</td>
<td>1397.0</td>
<td>0.204</td>
<td> (358.0/1755.0)</td></tr>
<tr><td>Total</td>
<td>4361.0</td>
<td>1661.0</td>
<td>0.1033</td>
<td> (622.0/6022.0)</td></tr></table></div>


    
**Maximum Metrics: Maximum metrics at their respective thresholds**
    



<div style="overflow:auto"><table style="width:50%"><tr><td><b>metric</b></td>
<td><b>threshold</b></td>
<td><b>value</b></td>
<td><b>idx</b></td></tr>
<tr><td>max f1</td>
<td>0.3</td>
<td>0.8</td>
<td>195.0</td></tr>
<tr><td>max f2</td>
<td>0.2</td>
<td>0.9</td>
<td>278.0</td></tr>
<tr><td>max f0point5</td>
<td>0.7</td>
<td>0.9</td>
<td>86.0</td></tr>
<tr><td>max accuracy</td>
<td>0.5</td>
<td>0.9</td>
<td>149.0</td></tr>
<tr><td>max precision</td>
<td>1.0</td>
<td>1.0</td>
<td>0.0</td></tr>
<tr><td>max absolute_MCC</td>
<td>0.3</td>
<td>0.7</td>
<td>195.0</td></tr>
<tr><td>max min_per_class_accuracy</td>
<td>0.2</td>
<td>0.9</td>
<td>247.0</td></tr></table></div>


    
    ModelMetricsBinomial: deeplearning
    ** Reported on validation data. **
    
    MSE: 0.0843305429737
    R^2: 0.593831388139
    LogLoss: 0.280203809486
    AUC: 0.930515181213
    Gini: 0.861030362427
    
**Confusion Matrix (Act/Pred) for max f1 @ threshold = 0.493462351545:**



<div style="overflow:auto"><table style="width:50%"><tr><td><b></b></td>
<td><b>false</b></td>
<td><b>true</b></td>
<td><b>Error</b></td>
<td><b>Rate</b></td></tr>
<tr><td>false</td>
<td>1361.0</td>
<td>62.0</td>
<td>0.0436</td>
<td> (62.0/1423.0)</td></tr>
<tr><td>true</td>
<td>158.0</td>
<td>435.0</td>
<td>0.2664</td>
<td> (158.0/593.0)</td></tr>
<tr><td>Total</td>
<td>1519.0</td>
<td>497.0</td>
<td>0.1091</td>
<td> (220.0/2016.0)</td></tr></table></div>


    
**Maximum Metrics: Maximum metrics at their respective thresholds**
    



<div style="overflow:auto"><table style="width:50%"><tr><td><b>metric</b></td>
<td><b>threshold</b></td>
<td><b>value</b></td>
<td><b>idx</b></td></tr>
<tr><td>max f1</td>
<td>0.5</td>
<td>0.8</td>
<td>137.0</td></tr>
<tr><td>max f2</td>
<td>0.1</td>
<td>0.8</td>
<td>303.0</td></tr>
<tr><td>max f0point5</td>
<td>0.7</td>
<td>0.9</td>
<td>82.0</td></tr>
<tr><td>max accuracy</td>
<td>0.7</td>
<td>0.9</td>
<td>91.0</td></tr>
<tr><td>max precision</td>
<td>1.0</td>
<td>1.0</td>
<td>0.0</td></tr>
<tr><td>max absolute_MCC</td>
<td>0.7</td>
<td>0.7</td>
<td>91.0</td></tr>
<tr><td>max min_per_class_accuracy</td>
<td>0.2</td>
<td>0.8</td>
<td>236.0</td></tr></table></div>


    
**Scoring History**:



<div style="overflow:auto"><table style="width:50%"><tr><td><b></b></td>
<td><b>timestamp</b></td>
<td><b>duration</b></td>
<td><b>training_speed</b></td>
<td><b>epochs</b></td>
<td><b>samples</b></td>
<td><b>training_MSE</b></td>
<td><b>training_r2</b></td>
<td><b>training_logloss</b></td>
<td><b>training_AUC</b></td>
<td><b>training_classification_error</b></td>
<td><b>validation_MSE</b></td>
<td><b>validation_r2</b></td>
<td><b>validation_logloss</b></td>
<td><b>validation_AUC</b></td>
<td><b>validation_classification_error</b></td></tr>
<tr><td></td>
<td>2015-11-06 17:57:05</td>
<td> 0.000 sec</td>
<td>None</td>
<td>0.0</td>
<td>0.0</td>
<td>nan</td>
<td>nan</td>
<td>nan</td>
<td>nan</td>
<td>nan</td>
<td>nan</td>
<td>nan</td>
<td>nan</td>
<td>nan</td>
<td>nan</td></tr>
<tr><td></td>
<td>2015-11-06 17:57:09</td>
<td> 2.899 sec</td>
<td>2594 rows/sec</td>
<td>1.0</td>
<td>6068.0</td>
<td>0.1</td>
<td>0.3</td>
<td>0.6</td>
<td>0.9</td>
<td>0.1</td>
<td>0.1</td>
<td>0.3</td>
<td>0.6</td>
<td>0.9</td>
<td>0.1</td></tr>
<tr><td></td>
<td>2015-11-06 17:57:15</td>
<td> 9.096 sec</td>
<td>5465 rows/sec</td>
<td>7.3</td>
<td>43742.0</td>
<td>0.1</td>
<td>0.6</td>
<td>0.3</td>
<td>0.9</td>
<td>0.1</td>
<td>0.1</td>
<td>0.6</td>
<td>0.3</td>
<td>0.9</td>
<td>0.1</td></tr>
<tr><td></td>
<td>2015-11-06 17:57:19</td>
<td>12.425 sec</td>
<td>6571 rows/sec</td>
<td>12.0</td>
<td>72478.0</td>
<td>0.1</td>
<td>0.6</td>
<td>0.2</td>
<td>1.0</td>
<td>0.1</td>
<td>0.1</td>
<td>0.6</td>
<td>0.3</td>
<td>0.9</td>
<td>0.1</td></tr></table></div>


    
**Variable Importances:**



<div style="overflow:auto"><table style="width:50%"><tr><td><b>variable</b></td>
<td><b>relative_importance</b></td>
<td><b>scaled_importance</b></td>
<td><b>percentage</b></td></tr>
<tr><td>Domestic.false</td>
<td>1.0</td>
<td>1.0</td>
<td>0.0</td></tr>
<tr><td>Primary_Type.NARCOTICS</td>
<td>0.9</td>
<td>0.9</td>
<td>0.0</td></tr>
<tr><td>IUCR.0860</td>
<td>0.8</td>
<td>0.8</td>
<td>0.0</td></tr>
<tr><td>FBI_Code.18</td>
<td>0.8</td>
<td>0.8</td>
<td>0.0</td></tr>
<tr><td>IUCR.4625</td>
<td>0.7</td>
<td>0.7</td>
<td>0.0</td></tr>
<tr><td>---</td>
<td>---</td>
<td>---</td>
<td>---</td></tr>
<tr><td>Location_Description.missing(NA)</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td></tr>
<tr><td>Primary_Type.missing(NA)</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td></tr>
<tr><td>FBI_Code.missing(NA)</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td></tr>
<tr><td>WeekDay.missing(NA)</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td></tr>
<tr><td>Domestic.missing(NA)</td>
<td>0.0</td>
<td>0.0</td>
<td>0.0</td></tr></table></div>





    




#31. Predict on the test set using the GBM model
    predictions = model_gbm.predict(test)
    predictions.show()


<table>
<tr><th>predict  </th><th style="text-align: right;">    false</th><th style="text-align: right;">      true</th></tr>
<tr><td>false    </td><td style="text-align: right;">0.946415 </td><td style="text-align: right;">0.0535847 </td></tr>
<tr><td>false    </td><td style="text-align: right;">0.862165 </td><td style="text-align: right;">0.137835  </td></tr>
<tr><td>false    </td><td style="text-align: right;">0.938661 </td><td style="text-align: right;">0.0613392 </td></tr>
<tr><td>false    </td><td style="text-align: right;">0.870186 </td><td style="text-align: right;">0.129814  </td></tr>
<tr><td>false    </td><td style="text-align: right;">0.980488 </td><td style="text-align: right;">0.0195118 </td></tr>
<tr><td>false    </td><td style="text-align: right;">0.972006 </td><td style="text-align: right;">0.0279937 </td></tr>
<tr><td>false    </td><td style="text-align: right;">0.990995 </td><td style="text-align: right;">0.00900489</td></tr>
<tr><td>true     </td><td style="text-align: right;">0.0210692</td><td style="text-align: right;">0.978931  </td></tr>
<tr><td>false    </td><td style="text-align: right;">0.693061 </td><td style="text-align: right;">0.306939  </td></tr>
<tr><td>false    </td><td style="text-align: right;">0.992097 </td><td style="text-align: right;">0.00790253</td></tr>
</table>



#32. Look at test set performance  (if it includes true labels)
    test_performance = model_gbm.model_performance(test)
    test_performance

    
    ModelMetricsBinomial: gbm
    ** Reported on test data. **
    
    MSE: 0.0893676876445
    R^2: 0.57094394422
    LogLoss: 0.294019576922
    AUC: 0.922152238508
    Gini: 0.844304477016
    
**Confusion Matrix (Act/Pred) for max f1 @ threshold = 0.365461652105:**



<div style="overflow:auto"><table style="width:50%"><tr><td><b></b></td>
<td><b>false</b></td>
<td><b>true</b></td>
<td><b>Error</b></td>
<td><b>Rate</b></td></tr>
<tr><td>false</td>
<td>1297.0</td>
<td>84.0</td>
<td>0.0608</td>
<td> (84.0/1381.0)</td></tr>
<tr><td>true</td>
<td>153.0</td>
<td>427.0</td>
<td>0.2638</td>
<td> (153.0/580.0)</td></tr>
<tr><td>Total</td>
<td>1450.0</td>
<td>511.0</td>
<td>0.1209</td>
<td> (237.0/1961.0)</td></tr></table></div>


    
**Maximum Metrics: Maximum metrics at their respective thresholds**
    



<div style="overflow:auto"><table style="width:50%"><tr><td><b>metric</b></td>
<td><b>threshold</b></td>
<td><b>value</b></td>
<td><b>idx</b></td></tr>
<tr><td>max f1</td>
<td>0.4</td>
<td>0.8</td>
<td>158.0</td></tr>
<tr><td>max f2</td>
<td>0.1</td>
<td>0.8</td>
<td>295.0</td></tr>
<tr><td>max f0point5</td>
<td>0.7</td>
<td>0.9</td>
<td>97.0</td></tr>
<tr><td>max accuracy</td>
<td>0.6</td>
<td>0.9</td>
<td>112.0</td></tr>
<tr><td>max precision</td>
<td>1.0</td>
<td>1.0</td>
<td>0.0</td></tr>
<tr><td>max absolute_MCC</td>
<td>0.6</td>
<td>0.7</td>
<td>112.0</td></tr>
<tr><td>max min_per_class_accuracy</td>
<td>0.2</td>
<td>0.8</td>
<td>235.0</td></tr></table></div>

#33. Create Plots of Crime type vs Arrest Rate and Proportion of reported Crime


    # Create table to report Crimetype, Arrest count per crime, total reported count per Crime  
    sqlContext.registerDataFrameAsTable(df_crimes, "df_crimes")
    allCrimes = sqlContext.sql("""SELECT Primary_Type, count(*) as all_count FROM df_crimes GROUP BY Primary_Type""")
    crimesWithArrest = sqlContext.sql("SELECT Primary_Type, count(*) as crime_count FROM chicagoCrime WHERE Arrest = 'true' GROUP BY Primary_Type")
    
    sqlContext.registerDataFrameAsTable(crimesWithArrest, "crimesWithArrest")
    sqlContext.registerDataFrameAsTable(allCrimes, "allCrimes")
    
    crime_type = sqlContext.sql("Select a.Primary_Type as Crime_Type, a.crime_count, b.all_count \
    FROM crimesWithArrest a \
    JOIN allCrimes b \
    ON a.Primary_Type = b.Primary_Type ")
    
    crime_type.show(12)

    +--------------------+-----------+---------+
    |          Crime_Type|crime_count|all_count|
    +--------------------+-----------+---------+
    |       OTHER OFFENSE|        183|      720|
    |   WEAPONS VIOLATION|         96|      118|
    |  DECEPTIVE PRACTICE|         25|      445|
    |            BURGLARY|         14|      458|
    |             BATTERY|        432|     1851|
    |             ROBBERY|         17|      357|
    | MOTOR VEHICLE THEFT|         17|      414|
    |        PROSTITUTION|        106|      106|
    |     CRIMINAL DAMAGE|         76|     1003|
    |          KIDNAPPING|          1|        7|
    |            GAMBLING|          3|        3|
    |LIQUOR LAW VIOLATION|         12|       12|
    +--------------------+-----------+---------+
    only showing top 12 rows
    



#34. Copy Crime_type table from Spark to H2O
    crime_typeHF = hc.as_h2o_frame(crime_type,framename="crime_type")


#35. Create Additional columns Arrest_rate and Crime_propotion 
    crime_typeHF["Arrest_rate"] = crime_typeHF["crime_count"]/crime_typeHF["all_count"]
    crime_typeHF["Crime_proportion"] = crime_typeHF["all_count"]/crime_typeHF["all_count"].sum()
    crime_typeHF["Crime_Type"] = crime_typeHF["Crime_Type"].asfactor()
    # h2o.assign(crime_typeHF,crime_type)
    crime_typeHF.frame_id = "Crime_type"


    crime_typeHF


<table>
<tr><th>Crime_Type         </th><th style="text-align: right;">  crime_count</th><th style="text-align: right;">  all_count</th><th style="text-align: right;">  Arrest_rate</th><th style="text-align: right;">  Crime_proportion</th></tr>
<tr><td>OTHER OFFENSE      </td><td style="text-align: right;">          183</td><td style="text-align: right;">        720</td><td style="text-align: right;">    0.254167 </td><td style="text-align: right;">       0.0721226  </td></tr>
<tr><td>WEAPONS VIOLATION  </td><td style="text-align: right;">           96</td><td style="text-align: right;">        118</td><td style="text-align: right;">    0.813559 </td><td style="text-align: right;">       0.0118201  </td></tr>
<tr><td>DECEPTIVE PRACTICE </td><td style="text-align: right;">           25</td><td style="text-align: right;">        445</td><td style="text-align: right;">    0.0561798</td><td style="text-align: right;">       0.0445758  </td></tr>
<tr><td>BURGLARY           </td><td style="text-align: right;">           14</td><td style="text-align: right;">        458</td><td style="text-align: right;">    0.0305677</td><td style="text-align: right;">       0.045878   </td></tr>
<tr><td>BATTERY            </td><td style="text-align: right;">          432</td><td style="text-align: right;">       1851</td><td style="text-align: right;">    0.233387 </td><td style="text-align: right;">       0.185415   </td></tr>
<tr><td>ROBBERY            </td><td style="text-align: right;">           17</td><td style="text-align: right;">        357</td><td style="text-align: right;">    0.047619 </td><td style="text-align: right;">       0.0357608  </td></tr>
<tr><td>MOTOR VEHICLE THEFT</td><td style="text-align: right;">           17</td><td style="text-align: right;">        414</td><td style="text-align: right;">    0.0410628</td><td style="text-align: right;">       0.0414705  </td></tr>
<tr><td>PROSTITUTION       </td><td style="text-align: right;">          106</td><td style="text-align: right;">        106</td><td style="text-align: right;">    1        </td><td style="text-align: right;">       0.0106181  </td></tr>
<tr><td>CRIMINAL DAMAGE    </td><td style="text-align: right;">           76</td><td style="text-align: right;">       1003</td><td style="text-align: right;">    0.0757727</td><td style="text-align: right;">       0.100471   </td></tr>
<tr><td>KIDNAPPING         </td><td style="text-align: right;">            1</td><td style="text-align: right;">          7</td><td style="text-align: right;">    0.142857 </td><td style="text-align: right;">       0.000701192</td></tr>
</table>





    




    hc


    H2OContext: ip=172.16.2.98, port=54329





    




#36. Plot in Flow 
    plot (g) -> g(
      g.rect(
        g.position "Crime_Type", "Arrest_rate"
        g.fillColor g.value 'blue'
        g.fillOpacity g.value 0.75
      )
      g.rect(
        g.position "Crime_Type", "Crime_proportion"
        g.fillColor g.value 'red'
        g.fillOpacity g.value 0.65
      )
      g.from inspect "data", getFrame "Crime_type"
    )


    #hc.stop()


    
