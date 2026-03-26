import json
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Optional

class FlightDetails(BaseModel):
    origin: str = Field(description="Origin city e.g. 'Denver'")
    destination: str = Field(description="Destination city e.g. 'Tokyo'")
    departure_date: Optional[str] = Field(default=None, description="Departure date e.g. 'June 15 2026'")
    return_date: Optional[str] = Field(default=None, description="Return date if round trip")

class HotelDetails(BaseModel):
    destination: str = Field(description="Destination city e.g. 'Tokyo'")
    departure_date: Optional[str] = Field(default=None, description="Check-in date e.g. 'June 5 2026'")
    return_date: Optional[str] = Field(default=None, description="Check-out date e.g. 'June 15 2026'")

class AttractionDetails(BaseModel):
    destination: str = Field(description="Destination city e.g. 'Tokyo'")
    interests: Optional[str] = Field(default=None, description="Traveler interests e.g. 'food, history, outdoor'")
    num_days: Optional[int] = Field(default=None, description="Number of days at destination — used to return enough attractions for the full trip")


@tool("find_flight_options", args_schema=FlightDetails)
def find_flight_options(origin: str, destination: str, departure_date: Optional[str] = None, return_date: Optional[str] = None) -> str:
    """ALWAYS use this tool to find flight options, airfare, or airline information. 
    Never answer flight questions or provide flight information from memory."""
    print(f"🔧 find_flight_options | {origin} → {destination} | {departure_date}")

    search = DuckDuckGoSearchRun()
    
    query = f"flights from {origin} to {destination}"
    if departure_date:
        query += f" {departure_date}"
    results = search.invoke(query)
    
    return json.dumps({
        "origin": origin,
        "destination": destination,
        "departure_date": departure_date,
        "return_date": return_date,
        "results": results,
        "instructions": "Extract airline names, prices, and booking websites from results. Format as a markdown table."
    })

@tool("find_hotel_options", args_schema=HotelDetails)
def find_hotel_options(destination: str, departure_date: Optional[str] = None, return_date: Optional[str] = None) -> str:
    """ALWAYS use this tool when the user asks about hotels, accommodation, 
    places to stay, or lodging. Never answer hotel questions from memory."""
    print(f"🔧 find_hotel_options | {destination} | {departure_date} → {return_date}")

    search = DuckDuckGoSearchRun()

    query = f"best hotels in {destination}"
    if departure_date and return_date:
        query += f" {departure_date} to {return_date}"
    results = search.invoke(query)
    return json.dumps({
        "destination": destination, 
        "departure_date": departure_date,
        "return_date": return_date,
        "instructions": "Extract hotel names, location, type (budget, kid friendly, luxury), booking websites, and quick description or value proposition from results. Include a variety of hotels in the area ranging in price and quality. Format as a markdown table.",
        "results": results})

@tool("find_nearby_attractions", args_schema=AttractionDetails)
def find_nearby_attractions(destination: str, interests: Optional[str] = None, num_days: Optional[int] = None) -> str:
    """ALWAYS use this tool to find attractions, things to do, or activities at a destination."""
    print(f"🔧 find_nearby_attractions | {destination} | {num_days} days with interest in {interests}")

    search = DuckDuckGoSearchRun()

    query = f"top attractions things to do in {destination}"
    if num_days:
        query += f" {num_days} day itinerary"
    if interests:
        query += f" for {interests}"
    results = search.invoke(query)
    return json.dumps({
        "destination": destination,
        "interests": interests, 
        "num_days": num_days, 
        "instructions": "Extract specific points of interest, popular attractions, activities or events include prices and booking websites from results if applicable. Format as a markdown table.",
        "results": results})