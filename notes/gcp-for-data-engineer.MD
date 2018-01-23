# Notes for Courera Course "GCP for Data Engineer"

## 1. GCP big data ml fundamentals

### Fundamentals of GCP
- computing: computing instance
- storage: google storage
	- edge points, data centers, regions
	- mulitple edge points for latecy
	- multiple zones (data centers) to avoid interruption
	- multiple regions for global access
	- gs - staging area for downstream analysis, e.g., dataproc, dataflow, sql, bigquery

### Typical users of GCP
- for migtration, e.g., hadoop, sql
	- dataproc, cloud-sql
	- need a hiearchy of different services, to provide step stones for different entry points to cloud
- for scalability and reliability
- for transition, greenfield customers, startup, innovations
	- serveless cloud service

### Resources
- cloud.google.com/pricing
- cloud.google.com/launcher

### Recommendataion use case
- a case showing a stepping stone from traditional projects to cloud
- use dataproc (hadoop) for batch processing, to update the recommendation scores
- save the model into a relationable table (cloud sql) for operational convience. 
- compared to other data access patterns, e.g., cloud storage, datastore, bigtable and bigquery, why is cloud sql the right pattern for this case?
- general data patterns for cloud sql:
	- create cloud sql instance
	- create database tables by importing .sql scripts from cloud storage
	- populate tables by importing .csv files from cloud storage (Remember cloud storage as a stagging area)
	- allow access to cloud sql
	- explore models using cloud shell
	- build models on hadoop using dataproc