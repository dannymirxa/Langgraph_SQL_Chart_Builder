from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi_app.models import Request, Response
from fastapi_app.service import main
from fastapi import Request as FastAPIRequest

router = APIRouter()
templates = Jinja2Templates(directory="templates")

@router.post("/request_insights")
async def request_insights(request: Request):
    # print("At post request", request)
    return await main(request)

@router.get("/get_chart", response_class=HTMLResponse)
async def get_chart(request: FastAPIRequest):
    return templates.TemplateResponse("index.html", {"request": request})
