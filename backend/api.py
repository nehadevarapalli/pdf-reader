import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, HTTPException, status, BackgroundTasks, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from io import BytesIO
import zipfile

from pipelines import standardize_docling, standardize_markitdown, html_to_md_docling, get_job_name, \
    pdf_to_md_docling, clean_temp_files

load_dotenv('../.env')
app = FastAPI()

@app.post("/processurl/", status_code=status.HTTP_200_OK)
async def process_url(
    background_tasks: BackgroundTasks,
    url: str, 
    include_markdown: bool = Query(False),
    include_images: bool = Query(False),
    include_tables: bool = Query(False),
    bundle: bool = Query(False)
):
    try:
        job_name = get_job_name()
        result = html_to_md_docling(url, job_name)
        background_tasks.add_task(my_background_task)

        if bundle:
            # Create a ZIP archive
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Markdown
                if include_markdown and result['markdown']:
                    zip_file.write(result['markdown'], arcname="document.md")

                # Images
                if include_images and result['images']:
                    for img in result['images'].iterdir():
                        zip_file.write(img, arcname=f"images/{img.name}")

                # Tables
                if include_tables and result['tables']:
                    for table in result['tables'].iterdir():
                        zip_file.write(table, arcname=f"tables/{table.name}")

            zip_buffer.seek(0)
            return StreamingResponse(
                zip_buffer,
                media_type='application/zip',
                headers={"Content-Disposition" : f"attachment; filename={job_name}.zip"}
            )
        else:
            if not include_markdown:
                raise HTTPException(status_code=400, detail="Only markdown output is supported for a non-bundled output")
            return FileResponse(
                result['markdown'],
                media_type="application/octet-stream",
                filename=f"{job_name}.md"
            )

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/processpdf/", status_code=status.HTTP_200_OK)
async def process_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile, 
    include_markdown: bool = Query(False),
    include_images: bool = Query(False),
    include_tables: bool = Query(False),
    bundle: bool = Query(False)
):
    if file.content_type != 'application/pdf':
        raise HTTPException(status_code=400, detail="File must be a PDF")

    background_tasks.add_task(my_background_task)
    contents = await file.read()
    output = Path("./temp_processing/output/pdf")
    os.makedirs(output, exist_ok=True)
    job_name = get_job_name()

    try:
        file_path = output / f'{job_name}.pdf'
        with open(file_path, 'wb') as f:
            f.write(contents)
            await file.close()

        result = pdf_to_md_docling(file_path, job_name)

        if bundle:
            # Create a ZIP archive
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Markdown
                if include_markdown and result['markdown']:
                    zip_file.write(result['markdown'], arcname="document.md")

                # Images
                if include_images and result['images']:
                    for img in result['images'].iterdir():
                        zip_file.write(img, arcname=f"images/{img.name}")

                # Tables
                if include_tables and result['tables']:
                    for table in result['tables'].iterdir():
                        zip_file.write(table, arcname=f"tables/{table.name}")

            zip_buffer.seek(0)
            return StreamingResponse(
                zip_buffer,
                media_type='application/zip',
                headers={"Content-Disposition" : f"attachment; filename={job_name}.zip"}
            )
        else:
            if not include_markdown:
                raise HTTPException(status_code=400, detail="Only markdown output is supported for a non-bundled output")

            return FileResponse(
                result['markdown'],
                media_type="application/octet-stream",
                filename=f"{job_name}.md"
            )
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        await file.close()

@app.post('/standardizedocling/', status_code=status.HTTP_200_OK)
async def standardizedocling(file: UploadFile, background_tasks: BackgroundTasks):
    if file.content_type != 'application/pdf':
        raise HTTPException(status_code=400, detail="File must be a PDF")
    background_tasks.add_task(my_background_task)
    contents = await file.read()
    output = Path("./temp_processing/output/pdf")
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
async def standardizemarkitdown(file: UploadFile, background_tasks: BackgroundTasks):
    if file.content_type != 'application/pdf':
        raise HTTPException(status_code=400, detail="File must be a PDF")
    background_tasks.add_task(my_background_task)
    contents = await file.read()
    output = Path("./temp_processing/output/pdf")
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
        await file.close()
    return FileResponse(standardized_output, media_type='application/octet-stream', filename=f'{file.filename}.md')


def my_background_task():
    clean_temp_files()
    print("Performed cleanup of temp files.")
