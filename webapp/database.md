```
sudo apt install mysql-server
pip install Flask mysql-connector-python

```
```
webapp/mysql_config.json
{
    "host": "localhost",
    "user": "user",
    "password": "password",
    "database": "easifa"
}
```

```
CREATE DATABASE easifa;
USE easifa;
CREATE TABLE qurey_data (
    uniprot_id VARCHAR(255),
    qurey_dataframe TEXT,
    message TEXT,
    calculated_sequence TEXT
);

CREATE TABLE predicted_results (
    uniprot_id VARCHAR(255),
    pred_active_site_labels JSON
);
```