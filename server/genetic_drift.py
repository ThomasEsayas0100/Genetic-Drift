import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import colorsys

def genetic_drift(p0=0.5, Ne=2000, nrep=5, time=100, show="p"):
    """
    Calculates the genetic drift over time.

    Args:
        p0 (float): The initial frequency of one allele in the population. Defaults to 0.5.
        Ne (int): The effective population size. Defaults to 2000.
        nrep (int): The number of replications. Defaults to 5.
        time (int): The number of time steps. Defaults to 100.
        show (str): The type of data to show. Options are "p" and "genotypes". Defaults to "p".

    Returns:
        tuple: A tuple containing the x and y values.
            x (ndarray): The x values.
            y (ndarray): The y values.
    """
    # Initialize empty lists and arrays
    x = []
    y = []
    genotypes = []
    freq = np.zeros((time, 3, nrep))
    p = np.zeros((time, nrep))

    np.random.seed(251)  # Set the random seed for reproducibility

    # Generate random genotypes for each replication
    for i in range(nrep):
        genotypes.append(np.random.choice([1, 0], size=(Ne, 2), p=[p0, 1 - p0]))

    # Calculate genotype frequencies for the first replication
    for i in range(nrep):
        counts = np.sum(genotypes[i], axis=1)
        hist = np.histogram(counts, bins=[-0.5, 0.5, 1.5, 2.5], density=True)
        freq[0, :, i] = hist[0]

    # Calculate mean genotype frequency for the first replication
    for i in range(nrep):
        p[0, i] = np.mean(genotypes[i])

    # Initialize X array with genotype frequencies for the first replication
    X = np.zeros((nrep, 3))
    for i in range(nrep):
        X[i, :] = freq[0, :, i]

    for i in range(1, time):
        new_gen = [np.random.choice(genotypes[j].flatten(), size=(Ne, 2)) for j in range(nrep)]
        genotypes = new_gen

        # Calculate genotype frequencies and mean genotype frequency for each replication
        for j in range(nrep):
            counts = np.sum(genotypes[j], axis=1)
            hist = np.histogram(counts, bins=[-0.5, 0.5, 1.5, 2.5], density=True)
            freq[i, :, j] = hist[0]
            p[i, j] = np.mean(genotypes[j])

        X = freq[i, :, :]

        # If show is "p", append mean genotype frequencies to y
        if show == "p":
            for j in range(nrep):
                y.append(p[i, j])

        # If show is "genotypes", append genotype frequencies to x and y
        elif show == "genotypes":
            for j in range(nrep):
                x.append(np.arange(3) + j * 3)
                y.append(X[:, j]) 

    # Reshape y for plotting if show is "p"
    if show == "p":
        y = np.insert(y, 0, np.ones(nrep) * p0)
        y = np.reshape(y, (time, nrep)).T
        x = np.linspace(0, time, np.shape(y)[1])

    # Reshape y for plotting if show is "genotypes"
    if show == "genotypes":
        y = np.array(y).T.reshape((3, time - 1, nrep))

    return x, y




def plot_genetic_drift(p0, Ne, nrep, time, show, time_filter, rep_filter):
    """
    Generates a plot to visualize the genetic drift over time.

    Parameters:
    - p0 (float): The initial allele frequency.
    - Ne (int): The effective population size.
    - nrep (int): The number of replications.
    - time (int): The number of generations.
    - show (str): The type of plot to show ("p" or "genotypes").
    - time_filter (Tuple[int]): The range of generations to include in the plot (start, end).
    - rep_filter (int): The replication number to include in the plot (-1 for all replications).

    Returns:
    - str: The HTML code for the generated plot.

    Note:
    - The function internally calls the `genetic_drift` function to obtain the data for the plot.
    - The function uses the Plotly library to create interactive plots.
    """
    Ne = int(Ne) # Population must be an integer
    x, y = genetic_drift(p0=p0, Ne=Ne, nrep=nrep, time=time, show=show)

    def filter_dataframe(df, time_filter, rep_filter):
            """
            Filter the DataFrame based on time_filter and rep_filter.
        
            Parameters:
            - df: pandas DataFrame
            - time_filter: tuple (start_time, end_time)
            - rep_filter: int
        
            Returns:
            - filtered_df: pandas DataFrame
            """
            # Check if rep_filter is not equal to -1
            if rep_filter != -1: # Then filter in only specific replication
                filtered_df = df[(df['y'] >= time_filter[0]) & (df['y'] <= time_filter[1]) & (df['nrep'] == rep_filter - 1)]
            else: # Filter all in all replications
                filtered_df = df[(df['y'] >= time_filter[0]) & (df['y'] <= time_filter[1])]
        
            return filtered_df
        
    if show == "p": 
        fig = go.Figure()
        # Colorblind-friendly colors
        colorscale = ['#9CFFD2', '#1E88E5', '#FFBF00', '#000000', '#E85656']
        # Loop through the replications
        for i in range(nrep):
            # Check if the replication should be shown based on the filter
            if i == (rep_filter - 1) or rep_filter == -1:
                fig.add_trace(go.Scatter(
                    x=x,
                    y=y[i],
                    name=f"Replication {i+1}",
                    line=dict(color=colorscale[i])
                ))

        fig.update_layout(
            xaxis_title="Generations",
            yaxis_title="Allele Frequency",
            title=dict(text="Allele Frequency over Time", font=dict(size=18)),
            template='plotly_white',
            yaxis_range=[0,1],
            xaxis_range=[time_filter[0], time_filter[1]],
            width=1000,
            height=500
        )




    elif show == "genotypes":
      dx = 0.8 / (nrep + 1)  # space between replications
      section_space = dx * (nrep + 2)  # space between genotypes
      genotypes = ['aa', 'Aa', 'AA'] 
      shades = { # shades for each replication  in each genotype
          genotypes[0]: ['rgba({}, {}, {}, 1)'.format(*colorsys.hsv_to_rgb(280/360, 1.0, 0.9 - i*(0.9/nrep))) for i in range(nrep)],
          genotypes[1]: ['rgba({}, {}, {}, 1)'.format(*colorsys.hsv_to_rgb(180/360, 1.0, 0.9 - i*(0.9/nrep))) for i in range(nrep)],
          genotypes[2]: ['rgba({}, {}, {}, 1)'.format(*colorsys.hsv_to_rgb(35/360, 1.0, 0.9 - i*(0.9/nrep))) for i in range(nrep)]
      }
      data = []
      x_pos = 0 # Current x position (along the Genotype (Replicates) axis)
      data = [] # Initialize an empty list to store data

      # Iterate over the genotypes
      for genotype in range(3):
          # Iterate over the generations
          for generation in range(time-1):
              # Iterate over the replications
              for replication in range(nrep):
                  # Create a dictionary with data
                  # x_pos: x position
                  # generation: generation number
                  # y: value from y matrix
                  # genotype: genotype name
                  # nrep: replication number
                  # shades: shade value
                  # symbols: list of symbols based on genotype
                  data.append({
                      'x': x_pos,
                      'y': generation,
                      'z': y[genotype][generation][replication],
                      'genotype': genotypes[genotype],
                      'nrep': replication,
                      'shades': shades[genotypes[genotype]][replication],
                      'symbols': ['circle', 'cross', 'square'][genotype]
                  })
                  # Update x_pos for the next replication
                  x_pos += dx
              # Reset x_pos for the next generation
              x_pos -= nrep * dx
          # Add section_space to x_pos for spacing between genotypes
          x_pos += section_space
      

      df = pd.DataFrame.from_dict(data) # Create a dataframe from the data
      # Filter the DataFrame based on time_filter and rep_filter
      def filter_dataframe(df, time_filter, rep_filter):
          """
          Filter the DataFrame based on time_filter and rep_filter.
      
          Parameters:
          - df: pandas DataFrame
          - time_filter: tuple (start_time, end_time)
          - rep_filter: int
      
          Returns:
          - filtered_df: pandas DataFrame
          """
          # Check if rep_filter is not equal to -1
          if rep_filter != -1: # Then filter in only specific replication
              filtered_df = df[(df['y'] >= time_filter[0]) & (df['y'] <= time_filter[1]) & (df['nrep'] == rep_filter - 1)]
          else: # Filter all in all replications
              filtered_df = df[(df['y'] >= time_filter[0]) & (df['y'] <= time_filter[1])]
      
          return filtered_df
      
      df = filter_dataframe(df, time_filter, rep_filter)

      # Add annotation (aa, Aa, or AA) for each genotype category
      annotations = [dict(
          x=(nrep * dx) / 2 + i * section_space,  
          y = -1 + min(df['y'].values),  
          z=0,  
          text=genotypes[i], 
          font=dict(
              color=['#b700ff', '#00eeff', '#ffae00'][i]
          ),
          showarrow=False, 
          yshift=-15  
      ) for i in range(3)]

      fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])
      
      # Add trace to the figure
      fig.add_trace(
          go.Scatter3d(
              x=df['x'],
              y=df['y'],
              z=df['z'],
              mode='markers',
              marker=dict(size=4),
              marker_color=df['shades'],
              marker_symbol=df['symbols']
          ),
          row=1,
          col=1
      )
      
      # Update the layout of the figure
      fig.update_layout(
          scene=dict(
              xaxis_title="Genotype (Replicates)",
              yaxis_title="Generation",
              zaxis_title="Frequency"
          ),
          template='plotly_white',
          width=750,
          height=750,
          title=dict(text="Genotype Frequencies over Time"),
          scene_annotations=annotations,
          scene_xaxis=dict(showticklabels=False) 
      )

    if show != "Select an Option":
      fig.update_layout(
        title_font=dict(size=18),
        xaxis=dict(title_font=dict(size=18)),
        yaxis=dict(title_font=dict(size=18)),
        font=dict(size=18),
        legend=dict(font=dict(size=18)),
        autosize=False,
      )
      
      return fig.to_html(full_html=False)