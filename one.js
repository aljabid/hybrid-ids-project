const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  HeadingLevel, AlignmentType, BorderStyle, WidthType, ShadingType,
  LevelFormat, PageNumber, PageBreak, VerticalAlign
} = require('docx');
const fs = require('fs');

const BLUE = "1A3C6B";
const BLUE_LIGHT = "D5E3F5";
const BLUE_MID = "2E5FA3";
const GRAY_LIGHT = "F3F4F6";
const GRAY_MID = "E5E7EB";
const ACCENT = "0F4C8A";
const RED = "C0392B";
const AMBER = "B7770D";
const GREEN_DARK = "1A6640";

const border = { style: BorderStyle.SINGLE, size: 1, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };
const noBorder = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const noBorders = { top: noBorder, bottom: noBorder, left: noBorder, right: noBorder };

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 360, after: 180 },
    children: [new TextRun({ text, bold: true, size: 36, color: BLUE, font: "Arial" })]
  });
}
function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 280, after: 140 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: BLUE_MID, space: 4 } },
    children: [new TextRun({ text, bold: true, size: 26, color: BLUE_MID, font: "Arial" })]
  });
}
function h3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 200, after: 100 },
    children: [new TextRun({ text, bold: true, size: 22, color: ACCENT, font: "Arial" })]
  });
}
function body(text, opts = {}) {
  return new Paragraph({
    spacing: { before: 60, after: 60 },
    children: [new TextRun({ text, size: 22, font: "Arial", ...opts })]
  });
}
function bullet(text, bold_prefix = null) {
  const runs = [];
  if (bold_prefix) {
    runs.push(new TextRun({ text: bold_prefix + " ", bold: true, size: 22, font: "Arial", color: ACCENT }));
    runs.push(new TextRun({ text, size: 22, font: "Arial" }));
  } else {
    runs.push(new TextRun({ text, size: 22, font: "Arial" }));
  }
  return new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    spacing: { before: 40, after: 40 },
    children: runs
  });
}
function subbullet(text) {
  return new Paragraph({
    numbering: { reference: "bullets2", level: 0 },
    spacing: { before: 20, after: 20 },
    children: [new TextRun({ text, size: 20, font: "Arial", color: "555555" })]
  });
}
function space(n = 1) {
  return new Paragraph({ spacing: { before: n * 60, after: 0 }, children: [new TextRun("")] });
}
function divider() {
  return new Paragraph({
    spacing: { before: 120, after: 120 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: GRAY_MID, space: 1 } },
    children: [new TextRun("")]
  });
}
function labeledCell(label, value, labelColor = BLUE_MID) {
  return new TableCell({
    borders,
    width: { size: 2340, type: WidthType.DXA },
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    children: [
      new Paragraph({ children: [new TextRun({ text: label, bold: true, size: 18, color: labelColor, font: "Arial" })] }),
      new Paragraph({ children: [new TextRun({ text: value, size: 20, font: "Arial" })] })
    ]
  });
}

function caseTable(fields) {
  // fields = [{label, value}] — 2-column layout
  const rows = [];
  for (let i = 0; i < fields.length; i += 2) {
    const cells = [
      new TableCell({
        borders,
        width: { size: 1500, type: WidthType.DXA },
        shading: { fill: BLUE_LIGHT, type: ShadingType.CLEAR },
        margins: { top: 80, bottom: 80, left: 120, right: 120 },
        children: [new Paragraph({ children: [new TextRun({ text: fields[i].label, bold: true, size: 19, font: "Arial", color: BLUE })] })]
      }),
      new TableCell({
        borders,
        width: { size: 3180, type: WidthType.DXA },
        margins: { top: 80, bottom: 80, left: 120, right: 120 },
        children: [new Paragraph({ children: [new TextRun({ text: fields[i].value, size: 20, font: "Arial" })] })]
      }),
    ];
    if (fields[i + 1]) {
      cells.push(new TableCell({
        borders,
        width: { size: 1500, type: WidthType.DXA },
        shading: { fill: BLUE_LIGHT, type: ShadingType.CLEAR },
        margins: { top: 80, bottom: 80, left: 120, right: 120 },
        children: [new Paragraph({ children: [new TextRun({ text: fields[i+1].label, bold: true, size: 19, font: "Arial", color: BLUE })] })]
      }));
      cells.push(new TableCell({
        borders,
        width: { size: 3180, type: WidthType.DXA },
        margins: { top: 80, bottom: 80, left: 120, right: 120 },
        children: [new Paragraph({ children: [new TextRun({ text: fields[i+1].value, size: 20, font: "Arial" })] })]
      }));
    }
    rows.push(new TableRow({ children: cells }));
  }
  return new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: fields.length > 2 ? [1500, 3180, 1500, 3180] : [1500, 7860],
    rows
  });
}

function featureTable(rows_data) {
  const headerRow = new TableRow({
    children: [
      new TableCell({ borders, width: { size: 2800, type: WidthType.DXA }, shading: { fill: BLUE, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, children: [new Paragraph({ children: [new TextRun({ text: "Feature", bold: true, size: 20, color: "FFFFFF", font: "Arial" })] })] }),
      new TableCell({ borders, width: { size: 2200, type: WidthType.DXA }, shading: { fill: BLUE, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, children: [new Paragraph({ children: [new TextRun({ text: "Derived From", bold: true, size: 20, color: "FFFFFF", font: "Arial" })] })] }),
      new TableCell({ borders, width: { size: 2400, type: WidthType.DXA }, shading: { fill: BLUE, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, children: [new Paragraph({ children: [new TextRun({ text: "Tech Stack", bold: true, size: 20, color: "FFFFFF", font: "Arial" })] })] }),
      new TableCell({ borders, width: { size: 1960, type: WidthType.DXA }, shading: { fill: BLUE, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, children: [new Paragraph({ children: [new TextRun({ text: "Priority", bold: true, size: 20, color: "FFFFFF", font: "Arial" })] })] }),
    ]
  });
  const tableRows = rows_data.map((r, i) => new TableRow({
    children: [
      new TableCell({ borders, width: { size: 2800, type: WidthType.DXA }, shading: { fill: i % 2 === 0 ? "FFFFFF" : GRAY_LIGHT, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, children: [new Paragraph({ children: [new TextRun({ text: r[0], bold: true, size: 19, font: "Arial" })] })] }),
      new TableCell({ borders, width: { size: 2200, type: WidthType.DXA }, shading: { fill: i % 2 === 0 ? "FFFFFF" : GRAY_LIGHT, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, children: [new Paragraph({ children: [new TextRun({ text: r[1], size: 19, font: "Arial", color: "444444" })] })] }),
      new TableCell({ borders, width: { size: 2400, type: WidthType.DXA }, shading: { fill: i % 2 === 0 ? "FFFFFF" : GRAY_LIGHT, type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, children: [new Paragraph({ children: [new TextRun({ text: r[2], size: 19, font: "Arial", color: "555555" })] })] }),
      new TableCell({ borders, width: { size: 1960, type: WidthType.DXA }, shading: { fill: r[3] === "Core MVP" ? "FFE8E8" : r[3] === "Extended" ? "FFF4D9" : "E8F4FF", type: ShadingType.CLEAR }, margins: { top: 80, bottom: 80, left: 120, right: 120 }, children: [new Paragraph({ children: [new TextRun({ text: r[3], bold: true, size: 19, font: "Arial", color: r[3] === "Core MVP" ? RED : r[3] === "Extended" ? AMBER : BLUE_MID })] })] }),
    ]
  }));
  return new Table({
    width: { size: 9360, type: WidthType.DXA },
    columnWidths: [2800, 2200, 2400, 1960],
    rows: [headerRow, ...tableRows]
  });
}

const doc = new Document({
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u2022", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 260 } } } }] },
      { reference: "bullets2", levels: [{ level: 0, format: LevelFormat.BULLET, text: "\u25E6", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 900, hanging: 260 } } } }] },
      { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 260 } } } }] },
    ]
  },
  styles: {
    default: { document: { run: { font: "Arial", size: 22 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 36, bold: true, font: "Arial", color: BLUE }, paragraph: { spacing: { before: 360, after: 180 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 26, bold: true, font: "Arial", color: BLUE_MID }, paragraph: { spacing: { before: 280, after: 140 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true, run: { size: 22, bold: true, font: "Arial", color: ACCENT }, paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 2 } },
    ]
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1260, bottom: 1440, left: 1260 }
      }
    },
    children: [
      // ── COVER ──
      space(4),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 0, after: 120 },
        children: [new TextRun({ text: "AFM AI HACKATHON 2026", bold: true, size: 52, color: BLUE, font: "Arial" })]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 0, after: 80 },
        border: { bottom: { style: BorderStyle.SINGLE, size: 10, color: BLUE_MID, space: 6 } },
        children: [new TextRun({ text: "Farabi Hub Square, Almaty  |  June 24, 2026", size: 24, color: "555555", font: "Arial" })]
      }),
      space(2),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 0, after: 60 },
        shading: { fill: BLUE, type: ShadingType.CLEAR },
        children: [new TextRun({ text: "  TRACK 1: DIGITAL SHADOW  ", bold: true, size: 44, color: "FFFFFF", font: "Arial" })]
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 40, after: 40 },
        children: [new TextRun({ text: "OSINT & DarkNet Monitoring", bold: true, size: 32, color: BLUE_MID, font: "Arial" })]
      }),
      space(2),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({ text: "Comprehensive Case Analysis  |  Prototype Feature Roadmap  |  Extended Vision", size: 22, color: "666666", font: "Arial" })]
      }),
      space(8),

      // ── EXECUTIVE SUMMARY ──
      h1("Executive Summary"),
      body("This document provides a complete intelligence base for building a Digital Shadow prototype — an OSINT and DarkNet monitoring solution targeting illicit financial activity in Kazakhstan and the wider CIS region. It draws exclusively on confirmed AFM investigations, enforcement actions, and judicial cases from 2022 to 2026."),
      space(),
      body("Kazakhstan's rapid emergence as Central Asia's largest crypto mining hub has created a shadow financial infrastructure of equal scale. The AFM has documented an escalating pattern: unlicensed exchanges, darknet laundering networks, ATM cash-out rings, shell company fraud, and illegal crypto mining — all requiring OSINT-grade intelligence capabilities to detect and disrupt."),
      space(),

      // Key stats table
      new Table({
        width: { size: 9360, type: WidthType.DXA },
        columnWidths: [3120, 3120, 3120],
        rows: [
          new TableRow({ children: [
            new TableCell({ borders, shading: { fill: BLUE, type: ShadingType.CLEAR }, margins: { top: 100, bottom: 100, left: 140, right: 140 }, width: { size: 3120, type: WidthType.DXA }, children: [
              new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "$224M+", bold: true, size: 40, color: "FFFFFF", font: "Arial" })] }),
              new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "DarkNet laundering (RAKS)", size: 19, color: "CCDDFF", font: "Arial" })] })
            ]}),
            new TableCell({ borders, shading: { fill: BLUE_MID, type: ShadingType.CLEAR }, margins: { top: 100, bottom: 100, left: 140, right: 140 }, width: { size: 3120, type: WidthType.DXA }, children: [
              new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "3,600+", bold: true, size: 40, color: "FFFFFF", font: "Arial" })] }),
              new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Shell companies (3 years)", size: 19, color: "CCDDFF", font: "Arial" })] })
            ]}),
            new TableCell({ borders, shading: { fill: ACCENT, type: ShadingType.CLEAR }, margins: { top: 100, bottom: 100, left: 140, right: 140 }, width: { size: 3120, type: WidthType.DXA }, children: [
              new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "$251M", bold: true, size: 40, color: "FFFFFF", font: "Arial" })] }),
              new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: "Illegal cash-out ops (2025)", size: 19, color: "CCDDFF", font: "Arial" })] })
            ]}),
          ]})
        ]
      }),
      space(2),

      // ── CASES ──
      h1("Documented Cases — Digital Shadow"),

      // CASE 1
      h2("Case 1 — RAKS Exchange: CIS-Wide DarkNet Laundering Hub"),
      caseTable([
        { label: "Date", value: "Sep 30, 2025" },
        { label: "Agency", value: "AFM Kazakhstan" },
        { label: "Amount Laundered", value: "$224 million USD" },
        { label: "Assets Frozen", value: "9.7 million USDT (67 wallets)" },
        { label: "Darknet Markets", value: "20+ marketplaces, 200+ drug shops" },
        { label: "Geographic Reach", value: "Kazakhstan, Russia, Ukraine, Moldova" },
      ]),
      space(),
      h3("What Happened"),
      body("RAKS was the largest shadow crypto exchange in the CIS region, operating for over three years as a professional money laundering service for darknet vendors. The platform serviced more than 200 drug marketplaces and online fraud networks with an estimated 5 million total users across the former Soviet space."),
      body("The AFM identified RAKS by monitoring suspicious crypto transaction patterns. Upon initiating a blockchain investigation, analysts traced over 4,000 cryptocurrency wallets and identified a cluster of addresses with clearly criminal proceeds. The platform began showing signs of a deliberate shutdown on September 29–30, prompting the AFM to act immediately — blocking 67 addresses and freezing 9.7 million USDT."),
      body("A subsequent seizure in April 2026 confirmed an additional 3.2 million USDT confiscated, bringing total frozen assets to over $12 million. The AFM also noted a wave of complaints from darknet vendors on criminal forums about RAKS not fulfilling its obligations — an intelligence signal that preceded the collapse."),
      h3("OSINT & DarkNet Signals That Were Detectable"),
      bullet("Suspicious on-chain transaction clustering across 4,000+ wallet addresses"),
      bullet("Vendor complaints appearing on darknet forums — a pre-collapse signal"),
      bullet("Sudden withdrawal of services from multiple connected platforms simultaneously"),
      bullet("Cross-jurisdiction money flows between KZ, RU, UA, MD with no legitimate commercial purpose"),
      bullet("USDT stablecoin concentration in wallets with no registered exchange activity"),
      bullet("Correlation between darknet market uptime and RAKS transaction volumes"),
      divider(),

      // CASE 2
      h2("Case 2 — 130 Unlicensed Crypto Exchanges: Multi-Step Laundering Network"),
      caseTable([
        { label: "Date", value: "Oct–Dec 2025" },
        { label: "Agency", value: "AFM Kazakhstan" },
        { label: "Platforms Closed", value: "130 unlicensed exchanges" },
        { label: "Assets Seized", value: "$16.7 million USD" },
        { label: "Laundering Volume", value: "60 billion KZT ($113M) in 2024 alone" },
        { label: "Geographic Reach", value: "KZ, Russia, Ukraine, Moldova" },
      ]),
      space(),
      h3("What Happened"),
      body("The AFM described a systematic multi-step laundering process used by these platforms: drug dealers and cybercriminals transferred proceeds (fiat or crypto) to exchange wallets, where funds were converted and routed through a rapid series of transactions to disguise origin. The exchanges acted as professional money launderers, cashing out and moving funds abroad."),
      body("In parallel, the AFM uncovered 81 shadow cash-out groups in 2024 with a combined turnover of 24 billion KZT ($43 million), using bank cards issued to nominal owners — so-called 'drop cards' — to facilitate anonymous ATM transactions. Cash withdrawals reached 13.2 trillion KZT ($24.1 billion), a trillion more than the previous year."),
      h3("OSINT & DarkNet Signals That Were Detectable"),
      bullet("Multi-hop transaction chains with no economic rationale"),
      bullet("Wallet addresses linked to both incoming drug proceeds and outgoing fiat conversion"),
      bullet("ATM withdrawal spikes correlated with drop card registration dates"),
      bullet("Shell company registrations clustered in specific geographic areas"),
      bullet("P2P exchange activity on platforms with no AIFC license"),
      divider(),

      // CASE 3
      h2("Case 3 — 3,600 Shell Companies: Money Laundering Corporate Infrastructure"),
      caseTable([
        { label: "Period", value: "3 years (to 2025)" },
        { label: "Agency", value: "AFM + Ministry of Justice + Ministry of AI" },
        { label: "Companies Cancelled", value: "3,600 shell companies" },
        { label: "Transactions Flagged", value: "30,000 fraudulent transactions" },
        { label: "Total Losses", value: "280 billion KZT ($511 million)" },
        { label: "New Measure", value: "Biometric verification for founders" },
      ]),
      space(),
      h3("What Happened"),
      body("The AFM, working with the Ministry of Justice and the Ministry of Artificial Intelligence, identified a massive infrastructure of fictitious companies used as laundering vehicles. Over three years, tax authorities cancelled 3,600 such registrations tied to 30,000 fraudulent transactions totalling 280 billion KZT ($511 million)."),
      body("The scheme involved registering companies using nominal owners — often people who had sold or rented their identity — then conducting transactions through them to layer proceeds. The AFM responded by drafting legislative amendments to require biometric verification of company founders and executives based on risk profiles."),
      h3("OSINT Signals That Were Detectable"),
      bullet("Company registrations with the same nominal founders appearing across multiple entities"),
      bullet("Businesses with no real commercial activity but high transaction volumes"),
      bullet("Founders matching known 'drop person' profiles (multiple registrations, no employment history)"),
      bullet("Company addresses clustered at known business registration services"),
      bullet("Transaction counterparties themselves having shell company characteristics"),
      divider(),

      // CASE 4
      h2("Case 4 — 15 Criminal Cash-Out Groups Dismantled (2025)"),
      caseTable([
        { label: "Date", value: "2025 (reported Jan 2026)" },
        { label: "Agency", value: "AFM Kazakhstan" },
        { label: "Groups Dismantled", value: "15 criminal groups" },
        { label: "Platforms Shut", value: "29 fiat conversion platforms" },
        { label: "Bank Cards Frozen", value: "20,000 drop cards" },
        { label: "Turnover", value: "128 billion KZT ($251 million)" },
      ]),
      space(),
      h3("What Happened"),
      body("The AFM confirmed the liquidation of 15 criminal networks operating platforms that converted digital funds to cash outside legal banking channels. In total 29 such platforms were shut down, with combined turnover of 128 billion KZT — representing a significant slice of Kazakhstan's shadow financial system."),
      body("The 20,000 frozen bank cards were 'drop cards' — accounts opened or surrendered by money mules who allowed criminal networks to use them for layering. The financial sector also terminated relationships with 2,000 companies and flagged 56,000 individuals suspected of money laundering activity."),
      h3("OSINT Signals That Were Detectable"),
      bullet("Bank account openings with nominal holders having no income or employment profile"),
      bullet("Rapid in-and-out transaction patterns on newly opened accounts"),
      bullet("Geographic clustering of money mules in specific districts"),
      bullet("Telegram/social media channels recruiting 'drop' account holders"),
      bullet("Correlation between account creation bursts and major fraud events"),
      divider(),

      // CASE 5
      h2("Case 5 — 24 Financial Pyramid Schemes with Crypto Proceeds (April 2025)"),
      caseTable([
        { label: "Date", value: "April 2025" },
        { label: "Agency", value: "AFM Kazakhstan" },
        { label: "Schemes Busted", value: "24 pyramid schemes" },
        { label: "Crypto Frozen", value: "$4.2 million" },
        { label: "Method", value: "Crypto conversion of pyramid proceeds" },
        { label: "Context", value: "Part of ongoing digital asset enforcement" },
      ]),
      space(),
      h3("What Happened"),
      body("In a single enforcement wave in April 2025, the AFM dismantled 24 financial pyramid schemes whose organizers converted proceeds into cryptocurrency to move and hide funds. $4.2 million in criminal crypto proceeds were frozen. This represented a significant acceleration in the AFM's digital asset investigation capacity."),
      h3("OSINT Signals That Were Detectable"),
      bullet("Rapid conversion of fiat investment proceeds into USDT or BTC on P2P platforms"),
      bullet("Social media recruitment posts correlating with wallet address creation dates"),
      bullet("Scheme websites registered within days of social media campaign launches"),
      bullet("Multiple schemes sharing the same crypto wallet infrastructure"),
      divider(),

      // CASE 6
      h2("Case 6 — Amir Capital: Cross-CIS Crypto Ponzi ($10M Seized)"),
      caseTable([
        { label: "Date", value: "September 2025" },
        { label: "Agency", value: "AFM Almaty Department" },
        { label: "Seized", value: "$10 million in crypto assets" },
        { label: "Victims", value: "KZ, Kyrgyzstan, Belarus, Russia" },
        { label: "Suspects", value: "40+ people implicated" },
        { label: "Cover Story", value: "International investment fund, Forex/crypto trading" },
      ]),
      space(),
      h3("What Happened"),
      body("Amir Capital presented itself as an international investment fund offering 5–10% monthly income from cryptocurrency trading, mining, and various crypto projects. It actively recruited victims across multiple CIS states, making it a cross-border fraud case. One of the principal organizers — a female suspect — was arrested and remanded in custody. Over 40 people were implicated."),
      body("The scheme operated for several years before detection, highlighting the challenge of monitoring investment fraud that spans multiple jurisdictions with no coordinated cross-border OSINT capability."),
      h3("OSINT Signals That Were Detectable"),
      bullet("Fake fund registration — company entity with no verifiable license in any CIS jurisdiction"),
      bullet("Multi-country social media recruitment campaigns with identical messaging"),
      bullet("Crypto trading claims with no auditable trading records"),
      bullet("Promised returns exceeding all legitimate investment benchmarks (5–10%/month)"),
      bullet("Referral networks expanding victim base across borders"),
      divider(),

      // CASE 7
      h2("Case 7 — 1,100+ Illegal Crypto Sites Blocked (Full-Year 2025)"),
      caseTable([
        { label: "Period", value: "Full year 2025 (reported Jan 2026)" },
        { label: "Agency", value: "AFM + National Security Committee + Ministry of Culture" },
        { label: "Sites Blocked", value: "1,100+ illegal trading services" },
        { label: "Criminal Cases", value: "1,135 completed" },
        { label: "Victim Compensation", value: "141.5 billion KZT ($277 million)" },
        { label: "Shell co. relationships terminated", value: "2,000 companies, 56,000 individuals flagged" },
      ]),
      space(),
      h3("What Happened"),
      body("In 2025 the AFM reported its largest-ever coordinated enforcement action: 1,100+ unlicensed crypto trading websites blocked, 1,135 criminal cases completed, and $277 million recovered for victims. Three agencies collaborated — AFM, the National Security Committee (NSC), and the Ministry of Culture and Information — reflecting the scale of the problem."),
      body("The multi-agency coordination required real-time intelligence sharing, OSINT domain monitoring, and rapid blocking capabilities that the current prototype can directly address."),
      h3("OSINT Signals That Were Detectable"),
      bullet("Domain registration patterns: bulk registration of similar-looking domains"),
      bullet("Hosting on servers shared with known illegal exchanges"),
      bullet("No AIFC license registration or regulatory filing"),
      bullet("Social media advertising to Kazakhstani users from unlicensed platforms"),
      bullet("Payment processor connections to blocked entities"),
      space(2),

      // ── ULTIMATE CASE STUDY ──
      h1("Ultimate Case Study: RAKS Exchange — Anatomy of a DarkNet Financial System"),
      new Paragraph({
        spacing: { before: 80, after: 80 },
        shading: { fill: BLUE_LIGHT, type: ShadingType.CLEAR },
        border: { left: { style: BorderStyle.SINGLE, size: 18, color: BLUE_MID } },
        children: [new TextRun({ text: "  Why RAKS is the definitive case: It represents the complete convergence of DarkNet markets, cross-jurisdiction crypto laundering, OSINT-detectable signals, and the intelligence methodology Kazakhstan will want to automate.", size: 21, font: "Arial", italics: true, color: BLUE })]
      }),
      space(),
      h3("The Criminal Architecture"),
      body("RAKS was not a simple exchange — it was a full-service financial infrastructure for CIS criminal networks. Its architecture had three layers:"),
      bullet("Intake layer: Drug dealers and online fraudsters from 200+ darknet shops deposited proceeds in fiat or crypto"),
      bullet("Conversion layer: RAKS converted and layered funds through rapid multi-hop transactions across 4,000+ wallet addresses"),
      bullet("Distribution layer: Clean funds exited to criminal operators in Kazakhstan, Russia, Ukraine, and Moldova"),
      space(),
      h3("How The AFM Detected It (The OSINT Methodology)"),
      body("The AFM's detection of RAKS followed what it described as a 'follow the money' approach on-chain:"),
      new Paragraph({ numbering: { reference: "numbers", level: 0 }, spacing: { before: 40, after: 40 }, children: [new TextRun({ text: "Anomaly detection: Automated monitoring of suspicious crypto transaction patterns flagged RAKS wallets", size: 22, font: "Arial" })] }),
      new Paragraph({ numbering: { reference: "numbers", level: 0 }, spacing: { before: 40, after: 40 }, children: [new TextRun({ text: "Graph expansion: AFM traced outgoing flows from flagged wallets, expanding to 4,000 linked addresses", size: 22, font: "Arial" })] }),
      new Paragraph({ numbering: { reference: "numbers", level: 0 }, spacing: { before: 40, after: 40 }, children: [new TextRun({ text: "DarkNet correlation: On-chain data matched to known darknet marketplace wallet clusters", size: 22, font: "Arial" })] }),
      new Paragraph({ numbering: { reference: "numbers", level: 0 }, spacing: { before: 40, after: 40 }, children: [new TextRun({ text: "Open-source intelligence: Vendor complaints on darknet forums about RAKS non-payment — a pre-collapse signal — were monitored", size: 22, font: "Arial" })] }),
      new Paragraph({ numbering: { reference: "numbers", level: 0 }, spacing: { before: 40, after: 40 }, children: [new TextRun({ text: "Timing intelligence: Platform shutdown behavior on Sep 29–30 triggered urgent freeze of 67 wallets before assets could be moved", size: 22, font: "Arial" })] }),
      space(),
      h3("What This Means For Your Prototype"),
      body("Every step in the AFM's RAKS investigation is automatable with AI. Your prototype can replicate and accelerate this detection pipeline, turning a months-long investigation into a near-real-time monitoring system."),
      space(2),

      // ── FEATURES ──
      h1("Prototype Feature Roadmap"),
      h2("Core MVP Features (Must demo on June 24)"),
      featureTable([
        ["Crypto Wallet Graph Clustering", "RAKS — 4,000 wallet analysis", "Python NetworkX / GraphX, Elliptic dataset, Chainalysis API", "Core MVP"],
        ["Anomalous Transaction Pattern Detection", "All 130 exchange + RAKS cases", "Isolation Forest, LSTM time-series, TensorFlow", "Core MVP"],
        ["USDT/Stablecoin Concentration Alerting", "Every KZ case — USDT is the dominant instrument", "Blockchain APIs (Tron/ETH), threshold rules", "Core MVP"],
        ["DarkNet Forum Signal Monitor", "RAKS pre-collapse vendor complaints", "Tor network crawler, NLP sentiment, Scrapy", "Core MVP"],
        ["Shell Company Identity Matching", "3,600 shell company investigation", "NER, company registry API, fuzzy matching", "Core MVP"],
      ]),
      space(),
      h2("Extended Features (Makes your solution impressive)"),
      featureTable([
        ["Drop Card / Money Mule Network Detection", "15 cash-out groups, 20,000 cards", "Graph anomaly detection, bank API patterns", "Extended"],
        ["Cross-Border Wallet Attribution (CIS-wide)", "RAKS multi-country operation", "Cross-chain analysis, heuristics", "Extended"],
        ["ATM Withdrawal Spike Correlation", "81 shadow cash-out groups, ATM vulnerability", "Geospatial analysis, time-series correlation", "Extended"],
        ["Multi-Exchange Layering Detection", "130 exchanges — multi-hop laundering", "Transaction graph path analysis, ML", "Extended"],
        ["P2P Platform Risk Scoring", "All pyramid cases — P2P conversion", "Platform API monitoring, risk ML model", "Extended"],
      ]),
      space(),
      h2("Futuristic Extended Services (Impress judges with vision)"),
      featureTable([
        ["Predictive Pre-Collapse Signal Engine", "RAKS vendor complaints as leading indicator", "NLP on darknet forums, anomaly trending", "Futuristic"],
        ["On-Chain + Off-Chain Identity Resolution", "RAKS operator anonymity problem", "ZK-proof resistant deanonymization, heuristics", "Futuristic"],
        ["CIS-Wide Intelligence Federation", "RAKS operated across 4 countries", "Federated learning, multi-agency API mesh", "Futuristic"],
        ["Real-Time Regulatory Blocking Recommendation", "1,100+ sites blocked with lag", "ML site classifier, automated submission API", "Futuristic"],
        ["AI-Generated Investigation Packages", "Manual AFM process takes months", "LLM report generation, evidence graph", "Futuristic"],
      ]),
      space(2),

      // ── PROTOTYPE ARCHITECTURE ──
      h1("Suggested Prototype Architecture"),
      h3("Data Ingestion Layer"),
      bullet("Public blockchain APIs: Etherscan, Tronscan, BscScan for USDT flows"),
      bullet("OSINT feeds: DarkNet forum crawlers (Tor), Telegram channel monitors"),
      bullet("Public company registries: Kazakhstan e-gov, cross-referenced with CIS registries"),
      bullet("Social media: VK, Instagram, TikTok, WhatsApp public channel scrapers"),
      space(),
      h3("Intelligence Processing Layer"),
      bullet("Graph neural network: wallet clustering and entity resolution (GraphSAGE or DGL)"),
      bullet("Anomaly detection: Isolation Forest / Autoencoder on transaction time-series"),
      bullet("NLP pipeline: multilingual (Kazakh/Russian/English) for darknet forum signals"),
      bullet("Risk scoring: composite model outputting 0–100 threat score per entity"),
      space(),
      h3("Output / Dashboard Layer"),
      bullet("Interactive graph visualization (D3.js or Gephi-export)"),
      bullet("Alert queue with investigation priority scoring"),
      bullet("Case-ready evidence package generator (auto-formatted for AFM submission)"),
      bullet("API endpoint for integration with existing AFM systems"),
      space(2),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { before: 200, after: 200 },
        shading: { fill: BLUE, type: ShadingType.CLEAR },
        children: [new TextRun({ text: "  AFM AI Hackathon 2026  |  Track 1: Digital Shadow  |  Almaty, June 24  ", bold: true, size: 22, color: "FFFFFF", font: "Arial" })]
      }),
    ]
  }]
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync('/home/claude/Track1_Digital_Shadow.docx', buf);
  console.log('Track 1 done');
});