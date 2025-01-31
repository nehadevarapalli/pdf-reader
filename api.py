import os
from pathlib import Path

from fastapi import FastAPI, UploadFile, HTTPException, status
from fastapi.responses import FileResponse
from pipelines import standardize_docling, standardize_markitdown, html_to_md_docling, get_job_name, output

app = FastAPI()

@app.post("/processurl/", status_code=status.HTTP_200_OK)
async def process_url(url: str):
    try:
        job_name = get_job_name()
        markdown_output = html_to_md_docling(url, job_name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return FileResponse(markdown_output, media_type='application/octet-stream',filename=f'{job_name}.md')


@app.post("/processpdf/", status_code=status.HTTP_200_OK)
async def process_pdf(url: str):
    # TODO: Implement this
    pass

@app.post('/standardizedocling/', status_code=status.HTTP_200_OK)
async def standardizedocling(file: UploadFile):
    contents = await file.read()
    output = Path("./output/pdf")
    os.makedirs(output, exist_ok=True)
    job_name = get_job_name()
    try:
        file_path = output / f'{job_name}.pdf'
        with open(file_path, 'wb') as f:
            f.write(contents)
            await file.close()
        standardized_output = standardize_docling(file_path, job_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await file.close()
    return FileResponse(standardized_output, media_type='application/octet-stream', filename=f'{file.filename}.md')

@app.post('/standardizemarkitdown/', status_code=status.HTTP_200_OK)
async def standardizemarkitdown(file: UploadFile):
    contents = await file.read()
    output = Path("./output/pdf")
    os.makedirs(output, exist_ok=True)
    job_name = get_job_name()
    try:
        file_path = output / f'{job_name}.pdf'
        with open(file_path, 'wb') as f:
            f.write(contents)
            await file.close()
        standardized_output = standardize_markitdown(file_path, job_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        file.file.close()
    return FileResponse(standardized_output, media_type='application/octet-stream', filename=f'{file.filename}.md')