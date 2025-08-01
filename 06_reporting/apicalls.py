import requests

#Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"



#Call each API endpoint and store the responses
response1 = requests.post(URL+'/prediction', data={'filepath': '/testdata/testdata.csv'}).content
response2 = requests.get(URL + '/scoring').content
response3 = requests.get(URL +'/summarystats').content
response4 = requests.get(URL + "/diagnostics").content

#combine all API responses
responses = {
    "prediction": response1,
    "scoring": response2,
    "summarystats": response3,
    "diagnostics": response4
}

#write the responses to your workspace



