# Whisper-WebUI REST API
REST API for Whisper-WebUI. Documentation is auto-generated upon deploying the app.
<br>[Swagger UI](https://github.com/swagger-api/swagger-ui) is available at `app/docs` or root URL with redirection. [Redoc](https://github.com/Redocly/redoc) is available at `app/redoc`.

## Configuration
You can set some server configurations in [config.yaml](https://github.com/jhj0517/Whisper-WebUI/blob/master/backend/configs/config.yaml). 
<br>For example, initial model size for Whisper or the cleanup frequency and TTL for cached files.
<br>If the endpoint generates and saves the file, all output files are stored in the `cache` directory, e.g. separated vocal/instrument files for `/bgm-separation` are saved in `cache` directory.

# Architecture

![diagram](https://github.com/user-attachments/assets/37d2ab2d-4eb4-4513-bb7b-027d0d631971)

The response can be obtained through [the polling API](https://docs.oracle.com/en/cloud/saas/marketing/responsys-develop/API/REST/Async/asyncApi-v1.3-requests-requestId-get.htm).
Each task is stored in the DB whenever the task is queued or updated by the process.

When the client first sends the `POST` request, the server returns an `identifier` to the client that can be used to track the status of the task. The task status is updated by the processes, and once the task is completed,  the client can finally obtain the result.

The client needs to implement manual API polling to do this, this is the example for the python client:
```python
def wait_for_task_completion(identifier: str,
                             max_attempts: int = 20,
                             frequency: int = 3) -> httpx.Response:
    """
    Polls the task status every `frequency` until it is completed, failed, or the `max_attempts` are reached.
    """
    attempts = 0
    while attempts < max_attempts:
        task = fetch_task(identifier)
        status = task.json()["status"]
        if status == "COMPLETED":
            return task["result"]
        if status == "FAILED":
            raise Exception("Task polling failed")
        time.sleep(frequency)
        attempts += 1
    return None
```
