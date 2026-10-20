"""Historical ONI episode membership, using the indexed NOAA snapshot.

NOAA historical convention: five consecutive seasons at rounded ONI >= 0.5 C.
Months use the central month of each season; MEI uses its ending month.
This is historical ONI classification, not the current operational RONI index.
"""

SEASONS = ('DJF','JFM','FMA','MAM','AMJ','MJJ','JJA','JAS','ASO','SON','OND','NDJ')
PERIODS = ('DEC-JAN','JAN-FEB','FEB-MAR','MAR-APR','APR-MAY','MAY-JUN',
           'JUN-JUL','JUL-AUG','AUG-SEP','SEP-OCT','OCT-NOV','NOV-DEC')
REFUSAL = ('The requested measurement and period are not fully available in the '
           'uploaded datasets. Please choose a period covered by that dataset.')


class EnsoScopeError(ValueError):
    pass


def stamp(record):
    d = record['data']
    sid = record['source_id']
    month = (d['month'] if sid == 'S67' else
             SEASONS.index(d['season']) + 1 if sid == 'S68' else
             PERIODS.index(d['period']) + 1)
    return d['year'] * 12 + month - 1


def episodes(records):
    rows = sorted((r for r in records if r.get('source_id') == 'S68'), key=stamp)
    runs, run = [], []
    previous = None
    for r in rows:
        point = stamp(r)
        if previous is not None and point != previous + 1:
            if len(run) >= 5:
                runs.append(run)
            run = []
        if round(float(r['data']['oni_anomaly_c']), 1) >= .5:
            run.append(r)
        else:
            if len(run) >= 5:
                runs.append(run)
            run = []
        previous = point
    if len(run) >= 5:
        runs.append(run)
    return runs


def select(records, scope):
    targets = {y * 12 + m - 1 for y in scope.years
               for m in (range(1, 13) if scope.entire_event else
                         scope.months or {SEASONS.index(s) + 1 for s in scope.seasons})}
    if not targets:
        raise EnsoScopeError(REFUSAL)
    selected = []
    for sid in scope.source_ids:
        available = {stamp(r): r for r in records if r.get('source_id') == sid}
        if not targets <= available.keys():
            raise EnsoScopeError(REFUSAL)
        selected.extend(available[t] for t in sorted(targets))
    return selected
