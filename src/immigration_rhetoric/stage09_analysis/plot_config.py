"""Centralised configuration for sentiment analysis visualisation."""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


class PlotConfig:
    """Configuration for plot appearance and behavior."""

    def __init__(
        self,
        figsize: Tuple[int, int] = (20, 8),
        dpi: int = 100,
        xlim: Optional[Tuple[str, str]] = None,
        legend_location: str = "upper left",
        legend_bbox: Tuple[float, float] = (0.0, 1.1),
        legend_columns: int = 6,
        grid_alpha: float = 0.3,
        bar_alpha: float = 0.7,
        line_width: int = 2,
    ):
        """Initialise plot configuration.

        Args:
            figsize: Figure size in inches (width, height)
            dpi: Dots per inch for output resolution
            xlim: Date range for x-axis as strings (start, end)
            legend_location: Location of legend
            legend_bbox: Bounding box for legend positioning
            legend_columns: Number of columns in legend
            grid_alpha: Transparency of grid lines
            bar_alpha: Transparency of bars
            line_width: Width of plot lines
        """
        self.figsize = figsize
        self.dpi = dpi
        self.xlim = xlim
        self.legend_location = legend_location
        self.legend_bbox = legend_bbox
        self.legend_columns = legend_columns
        self.grid_alpha = grid_alpha
        self.bar_alpha = bar_alpha
        self.line_width = line_width

    def create_figure(self, figsize: Optional[Tuple[int, int]] = None) -> Tuple[plt.Figure, plt.Axes]:  # type: ignore
        """Create a matplotlib figure with configured parameters."""
        return plt.subplots(figsize=figsize if figsize else self.figsize, dpi=self.dpi)

    def get_xlim_timestamps(
        self, default_x0: pd.Timestamp, default_x1: pd.Timestamp
    ) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """Get x-axis limits as timestamps."""
        if self.xlim:
            try:
                x0 = pd.Timestamp(self.xlim[0], tz="UTC")
                x1 = pd.Timestamp(self.xlim[1], tz="UTC")
                return x0, x1
            except Exception:
                return default_x0, default_x1
        return default_x0, default_x1


class DataConfig:
    """Configuration for data processing and analysis."""

    # Default parties of interest for analysis
    DEFAULT_PARTIES = [
        "Conservative Party",
        "Labour Party",
        "Liberal Democrats",
        "Reform UK",
    ]

    # Sentiment classification labels
    POSITIVE_LABEL = "POSITIVE_TOWARDS_IMMIGRATION"
    NEUTRAL_LABEL = "NEUTRAL_TOWARDS_IMMIGRATION"
    NEGATIVE_LABEL = "NEGATIVE_TOWARDS_IMMIGRATION"
    NOT_ABOUT_LABEL = "NOT_ABOUT_IMMIGRATION"

    # For each predicted label, the share of its snippets with each true label (see README)
    COUNT_CORRECTIONS = {
        POSITIVE_LABEL: {
            POSITIVE_LABEL: 0.6121,
            NEUTRAL_LABEL: 0.2182,
            NEGATIVE_LABEL: 0.1030,
            NOT_ABOUT_LABEL: 0.0667,
        },
        NEUTRAL_LABEL: {
            POSITIVE_LABEL: 0.1731,
            NEUTRAL_LABEL: 0.5659,
            NEGATIVE_LABEL: 0.1209,
            NOT_ABOUT_LABEL: 0.1401,
        },
        NEGATIVE_LABEL: {
            POSITIVE_LABEL: 0.0824,
            NEUTRAL_LABEL: 0.2747,
            NEGATIVE_LABEL: 0.5879,
            NOT_ABOUT_LABEL: 0.0549,
        },
        NOT_ABOUT_LABEL: {
            POSITIVE_LABEL: 0.0129,
            NEUTRAL_LABEL: 0.0426,
            NEGATIVE_LABEL: 0.0077,
            NOT_ABOUT_LABEL: 0.9368,
        },
    }

    # Party name standardisation mapping
    PARTY_MAPPING = {
        "Unionist Party": "Conservative Party",
        "Liberal Unionist Party": "Conservative Party",
        # "National Liberal Party": "Conservative Party",  # only valid after 1947
        "Scottish Labour Party": "Labour Party",
        "Labour Co-operative": "Labour Party",
        # "Social Democratic Party": "Liberal Democrats",
        # "Liberal Party": "Liberal Democrats",
        "Green Party of England and Wales": "Green Party",
        "UK Independence Party": "UKIP",
        "Scottish National Party": "SNP",
    }

    # Party plot time-windows
    PARTY_PLOT_WINDOWS = {
        "Scottish National Party": {
            "start": "1999-05-17"
        },  # Scottish devolved parliament established
        "UK Independence Party": {
            "start": "1993-09-03",
            "end": "2019-01-01",
        },  # UKIP active period
    }

    # Periods (start year, end year) for the ternary plots
    CUSTOM_PERIODS = [
        (1948, 1961),  # Post-war to Commonwealth Immigration Act
        (1962, 1980),  # Commonwealth Immigration Act to British Nationality Act
        (1981, 1991),  # British Nationality Act to Maastricht
        (1992, 2003),  # Maastricht to EU enlargement
        (2004, 2025),  # EU enlargement to present
    ]

    # Card et al. (2015) frames for plotting
    FRAMES = {
        "contributions": [
            "ability (n), attract (v), build (v), contribute (v),"
            "contribution (n), desirable (a), enlist (v), enrich (v),"
            "friendly (a), gain (v), genuine (a), hardworking (a), hero (n),"
            "highskilled (a), industrious (a), lawabide (v), loyal (a),"
            "patent (n), qualified (a), selfsufficiency (n), sturdy (a),"
            "veteran (n), worthy (a)"
        ],
        "crime": [
            "absconder (n), apprehend (v), arrest (n), arrest (v),"
            "bootlegger (n), capture (v), convict (v), criminal (a),"
            "criminal (n), criminalize (v), detain (v), detainee (n),"
            "felon (n), fugitive (a), gang (n), illegal (a), imprison (v),"
            "imprisonment (n), incarcerate (v), incarceration (n), inmate (n),"
            "jail (v), noncriminal (a), offense (n), parole (n), parole (v),"
            "prosecute (v), prosecution (n), smuggle (v), smuggler (n), smuggling (n), steal (v), terrorist (a), terrorist (n), unlawful (a), violent (a)"
        ],
        "culture": [
            "americanization (n), assimilate (v), assimilation (n),"
            "background (n), christian (a), citizenship (v), culture (n),"
            "diverse (a), diversity (n), ethnic (a), flag (n), foreigner (n),"
            "heritage (n), integrate (v), integration (n), language (n),"
            "nation (n), race (n), society (n), tradition (n)"
        ],
        "deficient": [
            "aidsinfected (a), cheap (a), diseased (a), drunk (a),"
            "ignorant (a), illiteracy (n), illiterate (a), insane (a),"
            "insane (n), objection- able (a), slacker (n), undesirable (a),"
            "unfit (a), unskilled (a)"
        ],
        "economic": [
            "agricultural (a), allocation (n), bank (n), budget (n), buy (v),"
            "cash (n), cent (n), coin (v), compete (v), competition (n),"
            "corporation (n), cost (v), economic (a), finance (v), fund (n),"
            "funding (n), investor (n), owner (n), ownership (n), pay (v),"
            "property (n), purchase (v), reimbursement (n), sale (n),"
            "specie (n), tax (n), tax (v), taxpaye (v), taxpayer (n),"
            "workforce (n)"
        ],
        "exclusion": [
            "admissibility (n), allow (v), ban (n), ban (v), bar (n), bar (v),"
            "cap (n), ceiling (n), criterion (n), debar (v), deny (v),"
            "deport (v), deportable (a), deportation (n), deported (a),"
            "detain (v), deter (v), eligibility (n), excludable (a),"
            "exclude (v), exclusion (n), expel (v), expulsion (n), forbid (v),"
            "inadmissible (a), inspect (v), interdict (v), interdiction (n),"
            "limit (n), limit (v), limitation (n), overstay (v), prevent (v),"
            "prohibit (v), prohibition (n), quota (n), reject (v),"
            "removable (a), removal (n), requirement (n), restrict (v),"
            "restriction (n), screen (v), screening (n), shut (v), stop (v),"
            "unauthorized (a), vet (v), vetting (n)"
        ],
        "family": [
            "ancestor (n), boy (n), child (n), daughter (n), descendant (n),"
            "descendent (n), family (n), familybased (a), familysponsored (a),"
            "generation (n), girl (n), grandchild (n), granddaughter (n),"
            "grandparent (n), grandson (n), household (n), husband (n),"
            "kid (n), marriage (n), marry (v), neighbor (n), neighborhood (n),"
            "orphan (n), parent (n), relative (a), relative (n), son (n),"
            "spouse (n), wife (n)"
        ],
        "flood/tide": [
            "absorb (v), absorption (n), drain (v), fill (v), flood (n),"
            "flood (v), flow (n), flow (v), inflow (n), influx (n),"
            "outflow (n), pour (v), spill (v), stream (n), stream (v),"
            "surge (n), tide (n), trickle (n), wave (n)"
        ],
        "labor": [
            "agricultural (a), compete (v), competition (n), crewman (n),"
            "employ (v), employee (n), employer (n), employment (n), "
            "employmentbased (a), farmworker (n), hire (v), hiring (n),"
            "laborer (n), miner (n), nurse (n), tailor (n), unskilled (a),"
            "worker (n), workforce (n)"
        ],
        "legality": [
            "adjudication (n), amnesty (n), applicant (n), application (n),"
            "authorization (n), authorize (v), citizenship (n),"
            "citizenship (v), eligibility (n), eligible (a), exemption (n),"
            "familysponsored (a), hearing (n), identification (n), illegal (a),"
            "inadmissible (a), ineligible (a), law (n), lawful (a), legal (a),"
            "legalization (n), legalize (v), legalized (a), legitimate (a),"
            "license (n), naturalization (n), naturalize (v), overstay (v),"
            "permanent (a), prosecute (v), qualify (v), registered (a),"
            "registration (n), sponsor (n), sponsor (v), status (n),"
            "unauthorized (a), undocumented (a), unlawful (a),"
            "unnaturalized (a), verification (n), visa (n)"
        ],
        "migration": [
            "arrival (n), arrive (v), boatload (n), come (v), coming (n),"
            "convoy (n), cross (v), depart (v), departure (n), depot (n),"
            "destination (n), displacement (n), emigrate (v), enter (v),"
            "entrance (n), entry (n), exodus (n), flight (n), immigrate (v),"
            "incoming (a), land (v), landing (n), leave (v), migrate (v),"
            "migration (n), move (v), movement (n), passage (n), passenger (n),"
            "port (n), reenter (v), relocate (v), relocation (n), resettle (v),"
            "resettlement (n), resettling (n), return (n), return (v),"
            "route (n), seaman (n), settle (v), settlement (n), ship (v),"
            "shipload (n), station (n), transport (v), travel (v)"
        ],
        "quantity": [
            "additional (a), boatload (n), bulk (n), bunch (n), cap (n),"
            "count (n), count (v), counting (n), crowd (n), entire (a),"
            "estimate (n), estimate (v), estimated (a), flock (v),"
            "fraction (n), horde (n), increase (v), majority (n), many (a),"
            "mass (n), masse (n), million (n), more (a), most (a), much (a),"
            "number (n), number (v), percentage (n), proportion (n),"
            "shipload (n), statistic (n), sum (n), thousand (n), total (a),"
            "total (n), twothird (n)"
        ],
        "threats": [
            "aidsinfected (a), anarchist (n), apprehension (n), attack (n),"
            "attack (v), belligerent (n), blame (v), combatant (n), danger (n),"
            "dangerous (a), diseased (a), enemy (n), gang (n), horde (n),"
            "infect (v), invasion (n), overrun (n), problem (n), terrorist (a),"
            "terrorist (n), threaten (v), violent (a)"
        ],
        "victims": [
            "acceptance (n), afflict (v), aid (n), aid (v), amnesty (n),"
            "assist (v), assistance (n), assisted (a), asylum (n),"
            "battered (a), compassion (n), deserve (v), desperate (a),"
            "escape (v), exploit (v), feed (v), flee (v), help (v),"
            "helpless (a), hope (v), hopeful (a), humanitarian (a),"
            "impoverished (a), languish (v), orphan (n), penniless (a),"
            "persecute (v), persecution (n), plight (n), prejudice (n),"
            "refuge (n), refugee (n), relief (n), rescue (v), reward (v),"
            "sanctuary (n), shelter (v), struggle (n), suffering (n),"
            "survivor (n), tragedy (n), unfortunate (a), victim (n),"
            "welcome (v)"
        ],
    }

    @classmethod
    def get_frames(cls) -> Dict[str, List[str]]:
        """Get Card et al. (2015) frames."""
        return {frame: terms.copy() for frame, terms in cls.FRAMES.items()}

    @classmethod
    def get_default_parties(cls) -> List[str]:
        """Get the default list of parties of interest."""
        return cls.DEFAULT_PARTIES.copy()

    @classmethod
    def get_party_windows(cls) -> dict:
        """Get the party plot time windows."""
        return cls.PARTY_PLOT_WINDOWS.copy()

    @classmethod
    def get_custom_periods(cls) -> List[Tuple[int, int]]:
        """Get the (start year, end year) periods for the ternary plots."""
        return cls.CUSTOM_PERIODS.copy()


class StyleConfig:
    """Configuration for visual styling and colours."""

    # Political party colour scheme
    PARTY_COLOURS = {
        "Labour Party": "#c60200",
        "Conservative Party": "#0177b6",
        "Liberal Democrats": "#df5e01",
        "Liberal Party": "#FFAC00",
        "Reform UK": "#3ebae2",
        "SNP": "#f5dc00",
        "Green Party": "#3cb53f",
        "Plaid Cymru": "#478e39",
        "DUP": "#8b0000",
        "Sinn Fein": "#22874c",
        "SDLP": "#24b4a9",
        "Alliance": "#caba19",
        "Independent": "#333333",
        "Other": "#767676",
    }

    # Other colours, using the colours from the Guardian design system, which
    # we can use for non-party elements
    FRAME_COLOURS = {
        "brand_pastel": "#506991",
        "highlight_dark": "#ffbb50",
        "news_pastel": "#ffbac8",
        "opinion_pastel": "#f9b376",
        "sport_pastel": "#90dcff",
        "culture_pastel": "#e7d4b9",
        "lifestyle_pastel": "#fec8d3",
    }

    # Styling constants
    ZERO_LINE_COLOUR = "#999999"
    ZERO_LINE_WIDTH = 1
    ZERO_LINE_ZORDER = 1

    EVENT_LINE_ALPHA = 0.7
    EVENT_LINE_STYLE = "--"
    EVENT_LINE_WIDTH = 1
    EVENT_TEXT_FONTSIZE = 10
    EVENT_TEXT_FONTWEIGHT = "bold"

    BASELINE_STYLE = "-"

    @classmethod
    def get_party_colour(cls, party_name: str) -> str:
        """Get colour for a political party."""
        return cls.PARTY_COLOURS.get(party_name, cls.PARTY_COLOURS["Other"])


class TernaryStyleConfig:
    """Configuration for the ternary plots' styling."""

    # Guardian design system main colours
    POSITIVE_COLOUR = "#bb3b80"  # Lifestyle
    NEUTRAL_COLOUR = "#a1845c"  # Culture
    NEGATIVE_COLOUR = "#e05e00"  # Opinion

    FIGSIZE = (12, 11)
    DPI = 300
    SCATTER_SIZE = 20
    SCATTER_ALPHA = 0.5

    CORNER_LABEL_FONTSIZE = 14
    CORNER_LABEL_OFFSET = 0.2
    CORNER_LABEL_FONTWEIGHT = "bold"

    # Number of terms to annotate near each corner and nearest the centre
    CORNER_ANNOTATION_COUNT = 6
    CENTRE_ANNOTATION_COUNT = 6
    CORNER_ANNOTATION_FONTSIZE = 9
    CENTRE_ANNOTATION_FONTSIZE = 8
    CORNER_ANNOTATION_COLOUR = "#111111"
    CENTRE_ANNOTATION_COLOUR = "#222222"

    # adjustText parameters, to stop annotations overlapping
    ADJUST_TEXT_EXPAND = (1.05, 1.2)
    ADJUST_TEXT_FORCE = 0.5
    # A fixed number of iterations, rather than adjustText's default time limit, so the layout is reproducible
    ADJUST_TEXT_ITER_LIM = 1000
    ADJUST_TEXT_ARROW_COLOUR = "#888888"
    ADJUST_TEXT_ARROW_WIDTH = 0.5
    ADJUST_TEXT_ARROW_ALPHA = 0.6


class AnalysisConfig:
    """High-level configuration combining all aspects."""

    def __init__(
        self,
        plot_config: Optional[PlotConfig] = None,
        data_config: Optional[DataConfig] = None,
    ):
        """Initialise analysis configuration.

        Args:
            plot_config: Plot-specific configuration
            data_config: Data processing configuration
        """
        self.plot = plot_config or PlotConfig()
        self.data = data_config or DataConfig()

    @classmethod
    def create_paper_plots(cls) -> "AnalysisConfig":
        """Create configuration used in the paper."""
        return cls(
            plot_config=PlotConfig(figsize=(20, 8), dpi=300, xlim=("1923", "2025"))
        )
