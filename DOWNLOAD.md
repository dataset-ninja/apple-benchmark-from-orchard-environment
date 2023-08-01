Dataset **Apple from Orchard Environment** can be downloaded in Supervisely format:

 [Download](https://assets.supervisely.com/supervisely-supervisely-assets-public/teams_storage/t/R/8j/upfoyMojmM8pweYzvkaeM6ODMdCEwL76uwsbr8HwscjCgP9laxMlN1UqQlnch3Liq6K8hchRbQ2hTBRXaUj8PM0SMaDpo0sjnzHYpThdHsAvtB8by50BcXZ42N41.tar)

As an alternative, it can be downloaded with *dataset-tools* package:
``` bash
pip install --upgrade dataset-tools
```

... using following python code:
``` python
import dataset_tools as dtools

dtools.download(dataset='Apple from Orchard Environment', dst_dir='~/dataset-ninja/')
```
The data in original format can be 🔗[downloaded here](https://github.com/dataset-ninja/apple-benchmark-from-orchard-environment)