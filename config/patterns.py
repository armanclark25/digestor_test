from typing import Dict, List, Optional
import re


class ExtractionPatterns:

    def __init__(self):
        self.patterns = self._load_default_patterns()
    
    def _load_default_patterns(self) -> Dict[str, List[str]]:
        return {
            'building_codes': self._get_building_code_patterns(),
            'material_specs': self._get_material_spec_patterns(),
            'project_info': self._get_project_info_patterns(),
            'dimensions': self._get_dimension_patterns(),
            'fire_protection': self._get_fire_protection_patterns(),
            'environmental_conditions': self._get_environmental_patterns(),
            'manufacturers': self._get_manufacturer_patterns(),
            'specification_structure': self._get_specification_patterns(),
            'quality_standards': self._get_quality_standards_patterns(),
            'structural_steel': self._get_structural_steel_patterns(),
            'welding_requirements': self._get_welding_patterns(),
            'connection_specs': self._get_connection_patterns(),
            'load_requirements': self._get_load_patterns(),
            'accessibility': self._get_accessibility_patterns(),
            'electrical': self._get_electrical_patterns(),
            'plumbing': self._get_plumbing_patterns(),
            'hvac': self._get_hvac_patterns(),
            'finishes': self._get_finish_patterns(),
            'safety_requirements': self._get_safety_patterns(),
            'testing_requirements': self._get_testing_patterns()
        }
    
    def _get_building_code_patterns(self) -> List[str]:
        return [
            r'(?:IBC|International Building Code)\s*(\d{4})',
            r'(?:OBC|Ohio Building Code)\s*(\d{4})',
            r'(?:AISI|ASTM|AWS|AISC|TMS)\s*([A-Z]?\s*\d+(?:[/-][A-Z]?\d+)*)',
            r'(?:NFPA|NEC)\s*(\d+)',
            r'(?:ANSI)\s*([A-Z]\d+(?:\.\d+)*)',
            r'(?:IECC|IFGC)\s*(?:(\d{4})\s*)?',
            r'Construction Type[:\s]*([A-Z0-9-]+)',
            r'Occupancy Classification[:\s]*([A-Z0-9-,\s]+)',
            r'(?:CBC|California Building Code)\s*(\d{4})',
            r'(?:UBC|Uniform Building Code)\s*(\d{4})',
            r'(?:IRC|International Residential Code)\s*(\d{4})',
            r'(?:IMC|International Mechanical Code)\s*(\d{4})',
            r'(?:IPC|International Plumbing Code)\s*(\d{4})',
            r'(?:IEBC|International Existing Building Code)\s*(\d{4})',
            r'ADA\s*(?:Requirements|Standards|Compliance)',
            r'(?:Fire Rating|Fire Resistance)[:\s]*(\d+(?:\.\d+)?)\s*(?:hour|hr)',
            r'Seismic\s*(?:Design|Category|Zone)[:\s]*([A-Z0-9]+)'
        ]
    
    def _get_material_spec_patterns(self) -> List[str]:
        return [
            r'(\d+)\s*(?:ga|gauge|gage)\b',
            r'(\d+(?:\.\d+)?)\s*(?:inch|in\.?|")\s*thick',
            r'Grade\s*([A-Z]?\d+)',
            r'Type\s*([A-Z]?\d+[A-Z]*)',
            r'ASTM\s*([A-Z]\s*\d+(?:[/-][A-Z]?\d+)*)',
            r'Thickness[:\s]*(\d+(?:\.\d+)?)\s*(?:inch|in\.?|mm)',
            r'Species[:\s]*([^\n,]+)',
            r'Temper\s*([A-Z]\d+)',
            r'Class\s*([A-Z0-9]+)',
            r'Finish[:\s]*([^\n,]+)',
            r'Color[:\s]*([^\n,]+)',
            r'Size[:\s]*(\d+(?:\.\d+)?(?:\s*x\s*\d+(?:\.\d+)?)*)',
            r'Nominal\s*Size[:\s]*([^\n,]+)',
            r'Weight[:\s]*(\d+(?:\.\d+)?)\s*(?:lb|kg|lbs)',
            r'Density[:\s]*(\d+(?:\.\d+)?)\s*(?:pcf|kg/m3)',
            r'R-Value[:\s]*(\d+(?:\.\d+)?)',
            r'U-Value[:\s]*(\d+(?:\.\d+)?)',
            r'STC\s*(?:Rating)?[:\s]*(\d+)',
            r'IIC\s*(?:Rating)?[:\s]*(\d+)'
        ]
    
    def _get_project_info_patterns(self) -> List[str]:
        return [
            r'Project[:\s]+([^\n]+)',
            r'(?:Location|Address)[:\s]+([^\n]+)',
            r'Date[:\s]+([^\n]+)',
            r'(?:Section|SECTION)\s+(\d+(?:\.\d+)*)\s*[-–]\s*([^\n]+)',
            r'Division\s+(\d+)\s*[-–]\s*([^\n]+)',
            r'Building\s*(?:Height|Area)[:\s]*([^\n]+)',
            r'Shell Permit[:\s]*([^\n]+)',
            r'(?:Tenant\s*Space|TENANT\s*SPACE)\s*([A-Z0-9]+)',
            r'Owner[:\s]*([^\n]+)',
            r'Architect[:\s]*([^\n]+)',
            r'Engineer[:\s]*([^\n]+)',
            r'Contractor[:\s]*([^\n]+)',
            r'Project\s*Number[:\s]*([^\n]+)',
            r'Drawing\s*Number[:\s]*([^\n]+)',
            r'Revision[:\s]*([^\n]+)',
            r'Sheet[:\s]*(\d+)\s*of\s*(\d+)',
            r'Scale[:\s]*([^\n]+)',
            r'Job\s*Number[:\s]*([^\n]+)'
        ]
    
    def _get_dimension_patterns(self) -> List[str]:
        return [
            r'(\d+(?:\.\d+)?)\s*(?:SF|sq\.?\s*ft\.?)',
            r'(\d+(?:\.\d+)?)\s*(?:ft|feet|\')\s*(?:x|\*|by)\s*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*(?:inches?|in\.?|")\s*(?:high|width|length|thick)',
            r'(\d+(?:\.\d+)?)\s*(?:mm)',
            r'Building\s*Area[:\s]*(\d+(?:,\d+)?)\s*SF',
            r'(\d+(?:\.\d+)?)\s*(?:inches?|in\.?|")\s*O\.?C\.?',
            r'Height[:\s]*(\d+(?:\.\d+)?)\s*(?:ft|feet|\')',
            r'Width[:\s]*(\d+(?:\.\d+)?)\s*(?:ft|feet|\'|in|")',
            r'Length[:\s]*(\d+(?:\.\d+)?)\s*(?:ft|feet|\'|in|")',
            r'Depth[:\s]*(\d+(?:\.\d+)?)\s*(?:ft|feet|\'|in|")',
            r'Diameter[:\s]*(\d+(?:\.\d+)?)\s*(?:ft|feet|\'|in|")',
            r'Radius[:\s]*(\d+(?:\.\d+)?)\s*(?:ft|feet|\'|in|")',
            r'Span[:\s]*(\d+(?:\.\d+)?)\s*(?:ft|feet|\')',
            r'Clear\s*(?:Height|Span)[:\s]*(\d+(?:\.\d+)?)\s*(?:ft|feet|\')'
        ]
    
    def _get_fire_protection_patterns(self) -> List[str]:
        return [
            r'Fire\s*(?:Rating|Resistance)[:\s]*(\d+(?:\.\d+)?)\s*(?:hour|hr)',
            r'Fire\s*Separation[:\s]*([^\n]+)',
            r'(?:Sprinkler|NFPA\s*13)[:\s]*([^\n]+)',
            r'Fire\s*Alarm[:\s]*([^\n]+)',
            r'Fire\s*Extinguisher[:\s]*([^\n]+)',
            r'Smoke\s*(?:Detector|Detection)[:\s]*([^\n]+)',
            r'Exit\s*(?:Sign|Light)[:\s]*([^\n]+)',
            r'Emergency\s*Light[:\s]*([^\n]+)',
            r'Fire\s*(?:Door|Wall|Barrier)[:\s]*([^\n]+)',
            r'Fire\s*Damper[:\s]*([^\n]+)',
            r'Standpipe[:\s]*([^\n]+)',
            r'Fire\s*Pump[:\s]*([^\n]+)',
            r'(?:UL|Underwriters\s*Laboratories)\s*([A-Z]?\d+)',
            r'Flame\s*Spread[:\s]*(\d+)',
            r'Smoke\s*Developed[:\s]*(\d+)'
        ]
    
    def _get_environmental_patterns(self) -> List[str]:
        return [
            r'(?:Dead|Live|Wind)\s*Load[:\s]*(\d+(?:\.\d+)?)\s*(?:psf|PSF|kPa)',
            r'(?:Temperature|Humidity)[:\s]*([^\n]+)',
            r'Deflection[:\s]*L\s*/\s*(\d+)',
            r'Environmental\s*conditions[:\s]*([^\n]+)',
            r'Snow\s*Load[:\s]*(\d+(?:\.\d+)?)\s*(?:psf|PSF|kPa)',
            r'Seismic\s*Load[:\s]*([^\n]+)',
            r'Wind\s*Speed[:\s]*(\d+(?:\.\d+)?)\s*(?:mph|kph)',
            r'Exposure\s*Category[:\s]*([A-Z])',
            r'(?:Design|Basic)\s*Wind\s*Speed[:\s]*(\d+(?:\.\d+)?)\s*(?:mph|kph)',
            r'Ground\s*Snow\s*Load[:\s]*(\d+(?:\.\d+)?)\s*(?:psf|PSF)',
            r'Frost\s*Line[:\s]*(\d+(?:\.\d+)?)\s*(?:ft|feet|in)',
            r'Soil\s*Bearing[:\s]*(\d+(?:\.\d+)?)\s*(?:psf|PSF|kPa)',
            r'Water\s*Table[:\s]*([^\n]+)'
        ]
    
    def _get_manufacturer_patterns(self) -> List[str]:
        return [
            r'ClarkDietrich',
            r'COLLABORATIVE DESIGN',
            r'(?:Manufacturer|MANUFACTURER)[:\s]*([^\n]+)',
            r'Product\s*Data[:\s]*([^\n]+)',
            r'Brand[:\s]*([^\n]+)',
            r'Model[:\s]*([^\n]+)',
            r'Catalog\s*(?:Number|No\.?)[:\s]*([^\n]+)',
            r'Part\s*(?:Number|No\.?)[:\s]*([^\n]+)',
            r'Series[:\s]*([^\n]+)',
            r'Supplier[:\s]*([^\n]+)',
            r'Vendor[:\s]*([^\n]+)',
            r'Source[:\s]*([^\n]+)'
        ]
    
    def _get_specification_patterns(self) -> List[str]:
        return [
            r'DIVISION\s+(\d+)\s*[-–—]\s*([^\n\r]+)',
            r'SECTION\s+(\d+(?:\s+\d+)*)\s*[-–—]\s*([^\n\r]+)',
            r'PART\s+([123])[:\s]*([^\n\r]+)',
            r'(?:SUMMARY|REFERENCES|SUBMITTALS|QUALITY ASSURANCE|PRODUCTS|EXECUTION)',
            r'(\d{2}\s+\d{2}\s+\d{2})\s*[-–—]\s*([^\n\r]+)',  # CSI format
            r'(?:Article|ARTICLE)\s+(\d+(?:\.\d+)*)[:\s]*([^\n\r]+)',
            r'(?:Paragraph|PARAGRAPH)\s+([A-Z]\.?)[:\s]*([^\n\r]+)',
            r'(?:Subparagraph|SUBPARAGRAPH)\s+(\d+\.?)[:\s]*([^\n\r]+)'
        ]
    
    def _get_quality_standards_patterns(self) -> List[str]:
        return [
            r'Quality\s*Assurance[:\s]*([^\n]+)',
            r'Comply\s*with[:\s]*([^\n]+)',
            r'Standard[:\s]*([^\n]+)',
            r'Warranty[:\s]*([^\n]+)',
            r'(?:Installation|INSTALLATION)\s*(?:method|instruction)[s]?[:\s]*([^\n]+)',
            r'Testing[:\s]*([^\n]+)',
            r'Inspection[:\s]*([^\n]+)',
            r'Certification[:\s]*([^\n]+)',
            r'Approval[:\s]*([^\n]+)',
            r'Acceptance[:\s]*([^\n]+)',
            r'Performance[:\s]*([^\n]+)',
            r'Tolerance[:\s]*([^\n]+)'
        ]
    
    def _get_structural_steel_patterns(self) -> List[str]:
        return [
            r'STRUCTURAL\s*STEEL',
            r'STEEL\s*(?:DECKING|DECK)',
            r'STEEL\s*TRUSS',
            r'COLD-FORMED\s*METAL\s*FRAMING',
            r'AISC["\s]*([^"\n]+)',
            r'ASTM\s*A(\d+)',
            r'AWS\s*(D\d+\.\d+)',
            r'STEEL\s*(?:STUD|FRAMING)',
            r'GALVANIZED\s*STEEL',
            r'HOT-DIP\s*GALVANIZED',
            r'(\d+)\s*GAUGE\s*STEEL',
            r'STEEL\s*(?:BEAM|COLUMN|JOIST)',
            r'W\s*(\d+)\s*[xX]\s*(\d+)',  # Wide flange
            r'HSS\s*(\d+(?:\.\d+)?)\s*[xX]\s*(\d+(?:\.\d+)?)',  # Hollow structural section
            r'L\s*(\d+(?:\.\d+)?)\s*[xX]\s*(\d+(?:\.\d+)?)',  # Angle
            r'C\s*(\d+(?:\.\d+)?)\s*[xX]\s*(\d+(?:\.\d+)?)',  # Channel
            r'ZINC\s*COATING\s*(G\d+)',
            r'YIELD\s*STRENGTH[:\s]*(\d+(?:\.\d+)?)\s*(?:ksi|MPa)',
            r'TENSILE\s*STRENGTH[:\s]*(\d+(?:\.\d+)?)\s*(?:ksi|MPa)'
        ]
    
    def _get_welding_patterns(self) -> List[str]:
        return [
            r'AWS\s+(D\d+\.\d+)',
            r'E\s*(\d+[xX]?)',
            r'(?:weld|WELD)[s]?\s+([^,\n]+)',
            r'(?:Fillet|FILLET)\s*(?:weld|WELD)[:\s]*([^,\n]+)',
            r'(?:Groove|GROOVE)\s*(?:weld|WELD)[:\s]*([^,\n]+)',
            r'(?:Plug|PLUG)\s*(?:weld|WELD)[:\s]*([^,\n]+)',
            r'(?:Spot|SPOT)\s*(?:weld|WELD)[:\s]*([^,\n]+)',
            r'Welding\s*Position[:\s]*([^,\n]+)',
            r'Welding\s*Process[:\s]*([^,\n]+)',
            r'(?:Root|ROOT)\s*(?:pass|opening)[:\s]*([^,\n]+)',
            r'(?:Backing|BACKING)[:\s]*([^,\n]+)',
            r'(?:Prequalified|PREQUALIFIED)\s*(?:weld|WELD)',
            r'Weld\s*Size[:\s]*(\d+(?:\.\d+)?)\s*(?:inch|in|mm)',
            r'Weld\s*Symbol[:\s]*([^,\n]+)'
        ]
    
    def _get_connection_patterns(self) -> List[str]:
        return [
            r'A325N?\s+([^,\n]+)',
            r'(\d+/\d+)["\s]*(?:diameter|dia)',
            r'(?:bolt|BOLT)[s]?\s+([^,\n]+)',
            r'(?:Rivet|RIVET)[s]?\s+([^,\n]+)',
            r'(?:Screw|SCREW)[s]?\s+([^,\n]+)',
            r'(?:Anchor|ANCHOR)\s*(?:bolt|rod)[s]?\s+([^,\n]+)',
            r'(?:Bearing|BEARING)\s*(?:type|connection)[:\s]*([^,\n]+)',
            r'(?:Slip|SLIP)\s*(?:critical|resistant)[:\s]*([^,\n]+)',
            r'(?:Pretension|PRETENSION)[:\s]*([^,\n]+)',
            r'(?:Torque|TORQUE)[:\s]*(\d+(?:\.\d+)?)\s*(?:ft-lb|N-m)',
            r'(?:Shear|SHEAR)\s*(?:strength|capacity)[:\s]*([^,\n]+)',
            r'(?:Tension|TENSION)\s*(?:strength|capacity)[:\s]*([^,\n]+)',
            r'Connection\s*(?:type|detail)[:\s]*([^,\n]+)'
        ]
    
    def _get_load_patterns(self) -> List[str]:
        return [
            r'(?:Dead|DEAD)\s*(?:load|LOAD)[:\s]*(\d+(?:\.\d+)?)\s*(?:psf|PSF|kPa)',
            r'(?:Live|LIVE)\s*(?:load|LOAD)[:\s]*(\d+(?:\.\d+)?)\s*(?:psf|PSF|kPa)',
            r'(?:Roof|ROOF)\s*(?:load|LOAD)[:\s]*(\d+(?:\.\d+)?)\s*(?:psf|PSF|kPa)',
            r'(?:Snow|SNOW)\s*(?:load|LOAD)[:\s]*(\d+(?:\.\d+)?)\s*(?:psf|PSF|kPa)',
            r'(?:Wind|WIND)\s*(?:load|LOAD)[:\s]*(\d+(?:\.\d+)?)\s*(?:psf|PSF|kPa)',
            r'(?:Seismic|SEISMIC)\s*(?:load|LOAD|force)[:\s]*([^,\n]+)',
            r'(?:Point|POINT)\s*(?:load|LOAD)[:\s]*(\d+(?:\.\d+)?)\s*(?:lb|lbs|kN)',
            r'(?:Distributed|DISTRIBUTED)\s*(?:load|LOAD)[:\s]*(\d+(?:\.\d+)?)\s*(?:plf|PLF|kN/m)',
            r'(?:Allowable|ALLOWABLE)\s*(?:load|stress)[:\s]*(\d+(?:\.\d+)?)\s*(?:psf|PSF|ksi|MPa)',
            r'Load\s*(?:factor|combination)[:\s]*([^,\n]+)',
            r'(?:Ultimate|ULTIMATE)\s*(?:load|strength)[:\s]*(\d+(?:\.\d+)?)',
            r'(?:Service|SERVICE)\s*(?:load|level)[:\s]*(\d+(?:\.\d+)?)'
        ]
    
    def _get_accessibility_patterns(self) -> List[str]:
        return [
            r'ADA\s*(?:Requirements|Compliance|Standards)',
            r'(?:Accessible|ACCESSIBLE)\s*(?:route|path|entrance)',
            r'(?:Barrier|BARRIER)\s*(?:free|removal)',
            r'ANSI\s*A117\.1',
            r'(?:Ramp|RAMP)\s*(?:slope|gradient)[:\s]*(\d+(?:\.\d+)?)\s*(?:%|percent)',
            r'(?:Handrail|HANDRAIL)[:\s]*([^,\n]+)',
            r'(?:Grab|GRAB)\s*(?:bar|rail)[:\s]*([^,\n]+)',
            r'(?:Door|DOOR)\s*(?:width|clearance)[:\s]*(\d+(?:\.\d+)?)\s*(?:inch|in|")',
            r'(?:Threshold|THRESHOLD)[:\s]*(\d+(?:\.\d+)?)\s*(?:inch|in|")',
            r'(?:Clear|CLEAR)\s*(?:width|space)[:\s]*(\d+(?:\.\d+)?)\s*(?:inch|in|")',
            r'(?:Accessible|ACCESSIBLE)\s*(?:parking|space)',
            r'(?:Van|VAN)\s*(?:accessible|space)',
            r'(?:Tactile|TACTILE)\s*(?:warning|surface)'
        ]
    
    def _get_electrical_patterns(self) -> List[str]:
        return [
            r'(?:Voltage|VOLTAGE)[:\s]*(\d+(?:\.\d+)?)\s*(?:V|volt)',
            r'(?:Amperage|AMPERAGE|Current)[:\s]*(\d+(?:\.\d+)?)\s*(?:A|amp)',
            r'(?:Wattage|WATTAGE|Power)[:\s]*(\d+(?:\.\d+)?)\s*(?:W|watt)',
            r'(?:Phase|PHASE)[:\s]*([13])\s*(?:phase|ph)',
            r'(?:Frequency|FREQUENCY)[:\s]*(\d+(?:\.\d+)?)\s*(?:Hz|hertz)',
            r'NEC\s*(?:Article|Section)\s*(\d+)',
            r'(?:Conduit|CONDUIT)[:\s]*([^,\n]+)',
            r'(?:Cable|CABLE)\s*(?:type|size)[:\s]*([^,\n]+)',
            r'(?:Outlet|OUTLET|Receptacle)[:\s]*([^,\n]+)',
            r'(?:Switch|SWITCH)[:\s]*([^,\n]+)',
            r'(?:Panel|PANEL|Panelboard)[:\s]*([^,\n]+)',
            r'(?:Circuit|CIRCUIT)\s*(?:breaker|protection)[:\s]*(\d+(?:\.\d+)?)\s*(?:A|amp)',
            r'(?:Ground|GROUND|Grounding)[:\s]*([^,\n]+)',
            r'(?:GFCI|GFI)\s*(?:protection|required)',
            r'(?:Emergency|EMERGENCY)\s*(?:power|lighting)'
        ]
    
    def _get_plumbing_patterns(self) -> List[str]:
        return [
            r'(?:Pipe|PIPE)\s*(?:size|diameter)[:\s]*(\d+(?:\.\d+)?)\s*(?:inch|in|")',
            r'(?:Water|WATER)\s*(?:pressure|supply)[:\s]*(\d+(?:\.\d+)?)\s*(?:psi|PSI)',
            r'(?:Drain|DRAIN|Sewer)[:\s]*([^,\n]+)',
            r'(?:Vent|VENT|Venting)[:\s]*([^,\n]+)',
            r'(?:Fixture|FIXTURE)[:\s]*([^,\n]+)',
            r'(?:Valve|VALVE)[:\s]*([^,\n]+)',
            r'(?:Faucet|FAUCET|Tap)[:\s]*([^,\n]+)',
            r'(?:Toilet|TOILET|Water closet)[:\s]*([^,\n]+)',
            r'(?:Sink|SINK|Lavatory)[:\s]*([^,\n]+)',
            r'(?:Shower|SHOWER|Tub)[:\s]*([^,\n]+)',
            r'IPC\s*(?:Section|Chapter)\s*(\d+)',
            r'(?:Hot|HOT)\s*(?:water|supply)[:\s]*(\d+(?:\.\d+)?)\s*(?:°F|F)',
            r'(?:Cold|COLD)\s*(?:water|supply)[:\s]*(\d+(?:\.\d+)?)\s*(?:°F|F)',
            r'(?:Backflow|BACKFLOW)\s*(?:prevention|preventer)[:\s]*([^,\n]+)',
            r'(?:Cleanout|CLEANOUT)[:\s]*([^,\n]+)'
        ]
    
    def _get_hvac_patterns(self) -> List[str]:
        return [
            r'(?:CFM|cfm)\s*(\d+(?:\.\d+)?)',
            r'(?:BTU|btu)[:\s]*(\d+(?:,\d+)?)',
            r'(?:Ton|TON|tons)[:\s]*(\d+(?:\.\d+)?)',
            r'(?:Temperature|TEMPERATURE)[:\s]*(\d+(?:\.\d+)?)\s*(?:°F|F|°C|C)',
            r'(?:Humidity|HUMIDITY)[:\s]*(\d+(?:\.\d+)?)\s*(?:%|percent)',
            r'(?:Static|STATIC)\s*(?:pressure|SP)[:\s]*(\d+(?:\.\d+)?)\s*(?:in|")',
            r'(?:Duct|DUCT|Ductwork)[:\s]*([^,\n]+)',
            r'(?:Filter|FILTER)[:\s]*([^,\n]+)',
            r'(?:Damper|DAMPER)[:\s]*([^,\n]+)',
            r'(?:VAV|Variable air volume)[:\s]*([^,\n]+)',
            r'(?:CAV|Constant air volume)[:\s]*([^,\n]+)',
            r'(?:AHU|Air handling unit)[:\s]*([^,\n]+)',
            r'(?:RTU|Rooftop unit)[:\s]*([^,\n]+)',
            r'(?:Heat|HEAT)\s*(?:pump|recovery)[:\s]*([^,\n]+)',
            r'(?:Chiller|CHILLER)[:\s]*([^,\n]+)',
            r'(?:Boiler|BOILER)[:\s]*([^,\n]+)'
        ]
    
    def _get_finish_patterns(self) -> List[str]:
        return [
            r'(?:Paint|PAINT)[:\s]*([^,\n]+)',
            r'(?:Stain|STAIN)[:\s]*([^,\n]+)',
            r'(?:Tile|TILE)[:\s]*([^,\n]+)',
            r'(?:Carpet|CARPET|Flooring)[:\s]*([^,\n]+)',
            r'(?:Vinyl|VINYL)[:\s]*([^,\n]+)',
            r'(?:Hardwood|HARDWOOD)[:\s]*([^,\n]+)',
            r'(?:Laminate|LAMINATE)[:\s]*([^,\n]+)',
            r'(?:Drywall|DRYWALL|Gypsum)[:\s]*([^,\n]+)',
            r'(?:Plaster|PLASTER)[:\s]*([^,\n]+)',
            r'(?:Ceiling|CEILING)[:\s]*([^,\n]+)',
            r'(?:Trim|TRIM|Molding)[:\s]*([^,\n]+)',
            r'(?:Baseboard|BASEBOARD)[:\s]*([^,\n]+)',
            r'(?:Door|DOOR)\s*(?:finish|hardware)[:\s]*([^,\n]+)',
            r'(?:Window|WINDOW)\s*(?:finish|trim)[:\s]*([^,\n]+)',
            r'(?:Cabinet|CABINET)[:\s]*([^,\n]+)'
        ]
    
    def _get_safety_patterns(self) -> List[str]:
        return [
            r'OSHA\s*(?:Standard|Regulation)\s*(\d+)',
            r'(?:Safety|SAFETY)\s*(?:requirement|standard)[:\s]*([^,\n]+)',
            r'(?:PPE|Personal protective equipment)[:\s]*([^,\n]+)',
            r'(?:Fall|FALL)\s*(?:protection|arrest)[:\s]*([^,\n]+)',
            r'(?:Guardrail|GUARDRAIL)[:\s]*([^,\n]+)',
            r'(?:Safety|SAFETY)\s*(?:net|netting)[:\s]*([^,\n]+)',
            r'(?:Hazardous|HAZARDOUS)\s*(?:material|substance)[:\s]*([^,\n]+)',
            r'(?:Ventilation|VENTILATION)\s*(?:requirement|rate)[:\s]*([^,\n]+)',
            r'(?:Noise|NOISE)\s*(?:level|limit)[:\s]*(\d+(?:\.\d+)?)\s*(?:dB|decibel)',
            r'(?:Lockout|LOCKOUT)\s*(?:tagout|procedure)[:\s]*([^,\n]+)',
            r'(?:Emergency|EMERGENCY)\s*(?:procedure|exit)[:\s]*([^,\n]+)',
            r'(?:First|FIRST)\s*(?:aid|medical)[:\s]*([^,\n]+)',
            r'(?:Training|TRAINING)\s*(?:requirement|certification)[:\s]*([^,\n]+)'
        ]
    
    def _get_testing_patterns(self) -> List[str]:
        return [
            r'(?:Test|TEST|Testing)[:\s]*([^,\n]+)',
            r'(?:Inspection|INSPECTION)[:\s]*([^,\n]+)',
            r'(?:Verification|VERIFICATION)[:\s]*([^,\n]+)',
            r'(?:Commissioning|COMMISSIONING)[:\s]*([^,\n]+)',
            r'(?:Performance|PERFORMANCE)\s*(?:test|testing)[:\s]*([^,\n]+)',
            r'(?:Load|LOAD)\s*(?:test|testing)[:\s]*([^,\n]+)',
            r'(?:Pressure|PRESSURE)\s*(?:test|testing)[:\s]*([^,\n]+)',
            r'(?:Leak|LEAK)\s*(?:test|testing)[:\s]*([^,\n]+)',
            r'(?:Fire|FIRE)\s*(?:test|testing)[:\s]*([^,\n]+)',
            r'(?:Seismic|SEISMIC)\s*(?:test|testing)[:\s]*([^,\n]+)',
            r'(?:Quality|QUALITY)\s*(?:control|assurance)[:\s]*([^,\n]+)',
            r'(?:Acceptance|ACCEPTANCE)\s*(?:test|criteria)[:\s]*([^,\n]+)',
            r'(?:Witness|WITNESS)\s*(?:test|point)[:\s]*([^,\n]+)',
            r'(?:Documentation|DOCUMENTATION)\s*(?:requirement|submittal)[:\s]*([^,\n]+)'
        ]
    
    def get_patterns(self, category: str = None) -> Dict[str, List[str]]:
        if category:
            return {category: self.patterns.get(category, [])}
        return self.patterns.copy()
    
    def add_pattern(self, category: str, pattern: str) -> None:
        if category not in self.patterns:
            self.patterns[category] = []
        self.patterns[category].append(pattern)
    
    def remove_pattern(self, category: str, pattern: str) -> bool:
        if category in self.patterns and pattern in self.patterns[category]:
            self.patterns[category].remove(pattern)
            return True
        return False
    
    def validate_pattern(self, pattern: str) -> bool:
        try:
            re.compile(pattern)
            return True
        except re.error:
            return False
    
    def get_categories(self) -> List[str]:
        return list(self.patterns.keys())
    
    def get_pattern_count(self, category: str = None) -> int:
        if category:
            return len(self.patterns.get(category, []))
        return sum(len(patterns) for patterns in self.patterns.values())


_extraction_patterns: Optional[ExtractionPatterns] = None


def get_patterns() -> ExtractionPatterns:
    global _extraction_patterns
    if _extraction_patterns is None:
        _extraction_patterns = ExtractionPatterns()
    return _extraction_patterns


def reset_patterns() -> None:
    global _extraction_patterns
    _extraction_patterns = None