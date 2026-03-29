from pydantic import BaseModel


class user_input(BaseModel):
    Administrative:int
    Administrative_Duration:float
    Informational:int
    Informational_Duration:float
    ProductRelated:int
    ProductRelated_Duration:float
    BounceRates:float
    ExitRates:float
    PageValues:float
    SpecialDay:float
    Month:object
    OperatingSystems:int
    Browser:int
    Region:int
    TrafficType:int
    VisitorType:str
    Weekend:bool
