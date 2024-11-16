#![allow(unused)]
use core::f64;
use image::{imageops::FilterType, DynamicImage, ImageBuffer, ImageReader, Rgb};
use rs_graph::{
	maxflow,
	traits::{DirectedEdge, Indexable},
	vecgraph::VecGraphBuilder,
	Builder, Graph, IndexGraph, VecGraph,
};
use spade::{handles::DirectedEdgeHandle, Triangulation};
use std::{
	collections::{HashMap, HashSet},
	env,
	mem::swap,
	time::Instant,
};

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tile {
	Water,
	TheEndOfTheFuckingWorld,
	Grass,
	Sand,
	Cliff,
}

impl Tile {
	fn from_color(color: &[u8]) -> Option<Tile> {
		use Tile::*;
		Some(match color[..3] {
			[33, 65, 74] | [49, 81, 90] => Water,
			[148, 117, 82] | [107, 109, 66] | [82, 105, 58] | [66, 101, 49] | [58, 97, 49] => Cliff,
			[r, g, b] if 5 * g as i32 > 4 * r as i32 && g > b => Grass,
			[r, g, b] if r > g && r > b => Sand,
			_ => return None,
		})
	}
}

type Capacity = i64;
const DEBUG: bool = false;
const CAPACITY_SCALE: f64 = 10.0;

fn main() -> std::io::Result<()> {
	let program_start_time = Instant::now();
	let args: Vec<String> = env::args().collect();
	assert!(args.len() == 3);

	let start = Instant::now();
	let img = ImageReader::open(&args[1])?
		.decode()
		.unwrap()
		// .resize(1920, 1080, FilterType::Nearest)
		.into_rgb8();
	let (w, h) = (img.width() as usize, img.height() as usize);
	let duration = start.elapsed();
	if DEBUG {
		eprintln!("Image loaded in {}ms", start.elapsed().as_millis());
	}

	let start = Instant::now();
	let pixels = img.clone().into_vec();
	let mut tile_map_grid_cells: Box<[Box<[Tile]>]> = (0..w)
		.map(|x| {
			let mut last_tile = Tile::TheEndOfTheFuckingWorld;
			(0..h)
				.map(|y| {
					let i = (x + y * w) * 3;
					let tile = Tile::from_color(&pixels[i..(i + 3)]).unwrap_or(last_tile);
					last_tile = tile;
					tile
				})
				.collect()
		})
		.collect();
	// for x in 0..w { for y in 0..h { grid[x][y] = pixels[x + y * w]; } }
	if DEBUG {
		eprintln!("Tiles parsed in {}ms", start.elapsed().as_millis());
	}

	let start = Instant::now();
	remove_unreachable(&mut tile_map_grid_cells);
	if DEBUG {
		eprintln!("Unreachable detected in {}ms", start.elapsed().as_millis())
	};

	let start = Instant::now();
	let grid_size = 32;
	let mut vertices: Vec<(i32, i32)> = vec![];
	{
		let d = (w.max(h).max(1000) / 20) as i32;
		let step = (d / 2) as usize;
		vertices.extend((0..w as i32).step_by(step).flat_map(|x| [(x, -d), (x, h as i32 - 1 + d)]));
		vertices.extend((0..h as i32).step_by(step).flat_map(|y| [(-d, y), (w as i32 - 1 + d, y)]));
	}
	for grid_x in (0..(w / grid_size)).map(|x| x * grid_size) {
		for grid_y in (0..(h / grid_size)).map(|y| y * grid_size) {
			let mut corners = [(0, 0); 8];
			let mut scores = [i32::MIN; 8];
			for x1 in (grid_x + 1)..(grid_x + grid_size - 1) {
				for y1 in (grid_y + 1)..(grid_y + grid_size - 1) {
					// (x1, y1) and (x2, y2) form water-ground edge
					if tile_map_grid_cells[x1][y1] != Tile::Water {
						continue;
					}
					for (corner_id, (dx, dy)) in
						[(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
							.into_iter()
							.enumerate()
					{
						let x2 = x1.wrapping_add(dx as usize);
						let y2 = y1.wrapping_add(dy as usize);
						if x2 >= w || y2 >= h || tile_map_grid_cells[x2][y2] == Tile::Water {
							continue;
						}
						let score = dx * x1 as i32 + dy * y1 as i32;
						if scores[corner_id] == i32::MIN
							|| score < scores[corner_id] || score == scores[corner_id]
							&& x1.abs_diff(grid_x + grid_size / 2)
								+ y1.abs_diff(grid_y + grid_size / 2)
								< (grid_x + grid_size / 2).abs_diff(corners[corner_id].0)
									+ (grid_y + grid_size / 2).abs_diff(corners[corner_id].1)
						{
							corners[corner_id] = (x1, y1);
							scores[corner_id] = score;
						}
					}
				}
			}
			let mut any_added = false;
			for (score, (x, y)) in scores.into_iter().zip(corners) {
				if score != i32::MIN
					&& (vertices.len() - vertices.len().min(corners.len())..vertices.len()).all(
						|i| {
							vertices[i].0.abs_diff(x as i32).max(vertices[i].1.abs_diff(y as i32))
								> 2
						},
					) {
					for i in (vertices.len() - vertices.len().min(corners.len())..vertices.len()) {}
					vertices.push((x as i32, y as i32));
					any_added = true;
				}
			}
			if !any_added
				&& (
					grid_x <= 0
						|| grid_y <= 0 || grid_x + 2 * grid_size > w
						|| grid_y + 2 * grid_size > h
					// || (grid_x / grid_size + grid_y / grid_size / 2) % 4 == 2
					// 	&& grid_y / grid_size % 4 == 0x
				) {
				let grid_size = grid_size as i32;
				let x = grid_x as i32 + grid_size / 2;
				let y = grid_y as i32 + grid_size / 2;
				vertices.push((x, y));
			}
		}
	}
	if DEBUG {
		eprintln!("Key points generated in {}ms", start.elapsed().as_millis())
	};

	let start = Instant::now();
	let mut edges = HashSet::new();
	// for (i, (x1, y1)) in (&vertices).into_iter().enumerate() {
	// 	let mut min_distance = usize::MAX;
	// 	let mut partner = 0;
	// 	for (j, (x2, y2)) in (&vertices).into_iter().enumerate() {
	// 		if x1 == x2 && y1 == y2 {
	// 			continue;
	// 		};
	// 		let distance = x1.abs_diff(*x2).pow(2) + y1.abs_diff(*y2).pow(2);
	// 		if distance < min_distance {
	// 			min_distance = distance;
	// 			partner = j;
	// 		}
	// 	}
	// 	let j = partner;
	// 	edges.insert(if i < j { (i, j) } else { (j, i) });
	// }
	let starting_position = (w as i32 / 2, h as i32 / 2);
	let mut starting_triangle = 0;
	let (triangles, paths) = boris_triangulation(&vertices);
	for (triangle_index, (u, v, w)) in triangles.iter().enumerate() {
		let mut triangle = [*u, *v, *w];
		triangle.sort();
		let (u, v, w) = (triangle[0], triangle[1], triangle[2]);
		edges.insert((u, v));
		edges.insert((v, w));
		edges.insert((u, w));
		if is_inside(starting_position, (vertices[u], vertices[v], vertices[w])) {
			// is_inside_verbose(starting_position, (vertices[u], vertices[v], vertices[w]));
			starting_triangle = triangle_index;
		}
	}
	if DEBUG {
		eprintln!("Triangulation generated in {}ms", start.elapsed().as_millis())
	};

	let start = Instant::now();
	let start_base_radius = 0.0625 * w.max(h) as f64;
	let mut graph = vec![Vec::with_capacity(3); triangles.len() + 2];
	for (u, vs) in paths.iter().enumerate() {
		let (t, verts) = (triangles[u], &vertices);
		let x = ((verts[t.0].0 + verts[t.1].0 + verts[t.2].0) / 3).clamp(0, w as i32 - 1);
		let y = ((verts[t.0].1 + verts[t.1].1 + verts[t.2].1) / 3).clamp(0, h as i32 - 1);
		for &v in vs {
			debug_assert!(u != v);
			if u > v {
				continue;
			}
			let (t, verts) = (triangles[v], &vertices);
			let x = ((verts[t.0].0 + verts[t.1].0 + verts[t.2].0) / 3).clamp(0, w as i32 - 1);
			let y = ((verts[t.0].1 + verts[t.1].1 + verts[t.2].1) / 3).clamp(0, h as i32 - 1);
			let u_verts = HashSet::from([triangles[u].0, triangles[u].1, triangles[u].2]);
			let v_verts = HashSet::from([triangles[v].0, triangles[v].1, triangles[v].2]);
			let edge: Vec<_> = u_verts.intersection(&v_verts).into_iter().collect();
			debug_assert!(edge.len() == 2);
			let (x1, y1) = vertices[*edge[0]];
			let (x2, y2) = vertices[*edge[1]];
			let (x0, y0) = starting_position;
			let bonus = ((w as f64 / 2.0).powi(2) + (h as f64 / 2.0).powi(2)).sqrt()
				/ (0..11)
					.map(|t| {
						(((x1 + (x2 - x1) * t / 10 - x0).pow(2)
							+ (y1 + (y2 - y1) * t / 10 - x0).pow(2)) as f64)
							.sqrt()
					})
					.min_by_key(|x| (1048576.0 * x) as i64)
					.unwrap();
			(((x1 - starting_position.0).pow(2) + (y1 - starting_position.1).pow(2)) as f64).sqrt();
			let (x1, y1, x2, y2) = (x1 as isize, y1 as isize, x2 as isize, y2 as isize);
			let edge_length = x1.abs_diff(x2).max(y1.abs_diff(y2)) as isize;
			let capacity = ((0..=edge_length)
				.map(|t| {
					let x = (x1 + ((x2 - x1) * t) / edge_length).clamp(0, w as isize - 1) as usize;
					let y = (y1 + ((y2 - y1) * t) / edge_length).clamp(0, h as isize - 1) as usize;
					return if tile_map_grid_cells[x][y] != Tile::Water { 1 } else { 0 };
				})
				.sum::<Capacity>() as f64
				/ (edge_length + 1) as f64
				* (((x2 - x1).pow(2) + (y2 - y1).pow(2)) as f64).sqrt()
				* CAPACITY_SCALE * bonus.min(10.0)) as Capacity;
			graph[u].push((v, capacity));
			graph[v].push((u, capacity));
		}
		if vs.len() < 3 {
			let v = graph.len() - 2;
			let capacity = (2 * (w + h)) as Capacity;
			graph[u].push((v, capacity));
			graph[v].push((u, capacity));
		}
		if (0..128).any(|i| {
			let alpha = i as f64 / 128.0 * std::f64::consts::TAU;
			let x = ((alpha.cos() * start_base_radius) + starting_position.0 as f64) as i32;
			let y = ((alpha.sin() * start_base_radius) + starting_position.1 as f64) as i32;
			let u_verts_positions =
				(vertices[triangles[u].0], vertices[triangles[u].1], vertices[triangles[u].2]);
			is_inside((x, y), u_verts_positions)
		}) {
			let v = graph.len() - 1;
			let capacity = (2 * (w + h)) as Capacity;
			graph[u].push((v, capacity));
			graph[v].push((u, capacity));
		}
	}
	if DEBUG {
		eprintln!("Graph generated in {}ms", start.elapsed().as_millis())
	};

	let start = Instant::now();
	let (graph_flow, flow_cut) = get_max_flow(&graph);
	let flow_cut: HashSet<_> = HashSet::from_iter(flow_cut.into_iter());
	let max_flow = graph_flow[graph_flow.len() - 1];
	if DEBUG {
		eprintln!("Max flow found in {}ms", start.elapsed().as_millis());
		eprintln!(
			"|V|={} |E|={} |T|={} f={}",
			vertices.len(),
			edges.len(),
			triangles.len(),
			max_flow
		);
	};

	let start = Instant::now();
	println!("{}", (max_flow as f64 / CAPACITY_SCALE).ceil() as i32);
	let mut output_image = img;
	for (u, vs) in graph.iter().enumerate() {
		if flow_cut.contains(&u) {
			continue;
		}
		for &(v, capacity) in vs {
			if capacity == 0 || !flow_cut.contains(&v) || v >= triangles.len() {
				continue;
			}

			let u_verts = HashSet::from([triangles[u].0, triangles[u].1, triangles[u].2]);
			let v_verts = HashSet::from([triangles[v].0, triangles[v].1, triangles[v].2]);
			let edge: Vec<_> = u_verts.intersection(&v_verts).into_iter().collect();
			debug_assert!(edge.len() == 2);
			let (x1, y1) = vertices[*edge[0]];
			let (x2, y2) = vertices[*edge[1]];

			let d = x1.abs_diff(x2).max(y1.abs_diff(y2)) as i32;
			if d as f64 * CAPACITY_SCALE > 5.0 * capacity as f64 {
				continue;
			}
			for (x, y) in (0..=d).map(|t| ((x1 + ((x2 - x1) * t) / d), (y1 + ((y2 - y1) * t) / d)))
			{
				for (dx, dy) in (-2..3).flat_map(|dx| (-2..3).map(move |dy| (dx, dy))) {
					let Some(pixel) =
						output_image.get_pixel_mut_checked((x + dx) as u32, (y + dy) as u32)
					else {
						continue;
					};
					let (mut r, mut g, mut b) = (pixel.0[0], pixel.0[1], pixel.0[2]);
					r = r.saturating_add(38);
					b = b.saturating_add(38);
					*pixel = Rgb([r, g, b]);
				}
			}
		}
	}
	output_image.save(&args[args.len() - 1]);
	if DEBUG {
		output_image.save("factorio-map-processor-debug-6.png").unwrap();
		eprintln!("Output generated in {}ms", start.elapsed().as_millis());
	}

	if DEBUG {
		let start = Instant::now();
		let mut debug_image = ImageBuffer::<Rgb<u8>, Vec<_>>::new(w as u32, h as u32);
		for (x, y, pixel) in debug_image.enumerate_pixels_mut() {
			use Tile::*;
			*pixel = Rgb(match tile_map_grid_cells[x as usize][y as usize] {
				TheEndOfTheFuckingWorld => [255, 0, 255],
				Water => [33, 65, 74],
				Grass => [66, 57, 8],
				Cliff => [148, 117, 82],
				Sand => [99, 69, 33],
			});
			if x as usize % grid_size == 0 || y as usize % grid_size == 0 {
				*pixel = Rgb([0, 0, 0]);
			}
		}
		debug_image.save("factorio-map-processor-debug-1.png").unwrap();
		for (x, y) in &vertices {
			let x = *x as i32;
			let y = *y as i32;
			let point_radius = 2;
			for dy in -point_radius..=point_radius {
				for dx in -point_radius..=point_radius {
					let Some(pixel) = debug_image
						.get_pixel_mut_checked((x as i32 + dx) as u32, (y as i32 + dy) as u32)
					else {
						continue;
					};
					let d = dx.abs().max(dy.abs()) as u8 * (255 / point_radius.max(1) as u8);
					let alpha = 255 - ((d as i32).pow(2) / 255) as u8;
					*pixel = Rgb([255, alpha, alpha]);
				}
			}
		}
		debug_image.save("factorio-map-processor-debug-2.png").unwrap();
		for (u, v) in &edges {
			let (x1, y1) = vertices[*u];
			let (x2, y2) = vertices[*v];
			let (x1, y1, x2, y2) = (x1 as isize, y1 as isize, x2 as isize, y2 as isize);
			let d = x1.abs_diff(x2).max(y1.abs_diff(y2)) as isize;
			for (x, y) in (0..=d).flat_map(|t| {
				[
					((x1 + ((x2 - x1) * t) / d), (y1 + ((y2 - y1) * t) / d)),
					// ((x1 + ((x2 - x1) * t + d - 1) / d), (y1 + ((y2 - y1) * t + d - 1) / d)),
					// ((x1 + ((x2 - x1) * t + d / 2) / d), (y1 + ((y2 - y1) * t + d / 2) / d)),
				]
			}) {
				let Some(pixel) = debug_image.get_pixel_mut_checked(x as u32, y as u32) else {
					continue;
				};
				let (mut r, mut g, mut b) = (pixel.0[0], pixel.0[1], pixel.0[2]);
				let edge_brightness = 96;
				r = r.saturating_add(edge_brightness);
				g = g.saturating_add(edge_brightness);
				b = b.saturating_add(edge_brightness);
				*pixel = Rgb([r, g, b]);
			}
		}
		debug_image.save("factorio-map-processor-debug-3.png").unwrap();
		{
			for (i, &(v1, v2, v3)) in triangles.iter().enumerate() {
				if !graph[i].iter().any(|&(u, _)| u == graph.len() - 1) {
					continue;
				}
				let (x1, y1) = vertices[v1];
				let (x2, y2) = vertices[v2];
				let (x3, y3) = vertices[v3];
				for y in y1.min(y2).min(y3)..=y1.max(y2).max(y3) {
					for x in x1.min(x2).min(x3)..=x1.max(x2).max(x3) {
						if is_inside((x, y), ((x1, y1), (x2, y2), (x3, y3))) {
							let Some(pixel) = debug_image.get_pixel_mut_checked(x as u32, y as u32)
							else {
								continue;
							};
							let (mut r, mut g, mut b) = (pixel.0[0], pixel.0[1], pixel.0[2]);
							let edge_brightness = 32;
							r = r.saturating_add(edge_brightness);
							g = g.saturating_add(edge_brightness);
							b = b.saturating_add(edge_brightness);
							*pixel = Rgb([r, g, b]);
						}
					}
				}
			}
			for dy in -2..=2 {
				for dx in -2..=2 {
					*debug_image.get_pixel_mut(
						(starting_position.0 as i32 + dx) as u32,
						(starting_position.1 as i32 + dy) as u32,
					) = Rgb([192, 255, 255]);
				}
			}
			for i in 0..128 {
				let alpha = i as f64 / 128.0 * std::f64::consts::TAU;
				let x = ((alpha.cos() * start_base_radius) + starting_position.0 as f64) as usize;
				let y = ((alpha.sin() * start_base_radius) + starting_position.1 as f64) as usize;
				for dy in -2..=2 {
					for dx in -2..=2 {
						*debug_image
							.get_pixel_mut((x as i32 + dx) as u32, (y as i32 + dy) as u32) = Rgb([192, 255, 255]);
					}
				}
			}
		}
		debug_image.save("factorio-map-processor-debug-4.png").unwrap();

		for (i, &(v1, v2, v3)) in triangles.iter().enumerate() {
			let (x1, y1) = vertices[v1];
			let (x2, y2) = vertices[v2];
			let (x3, y3) = vertices[v3];
			let perimiter = ((x2.abs_diff(x1).pow(2) + y2.abs_diff(y1).pow(2)) as f64).sqrt()
				+ ((x3.abs_diff(x1).pow(2) + y3.abs_diff(y1).pow(2)) as f64).sqrt()
				+ ((x2.abs_diff(x3).pow(2) + y2.abs_diff(y3).pow(2)) as f64).sqrt();
			let relative_flow = graph_flow[i] as f64 / perimiter / CAPACITY_SCALE;
			if relative_flow == 0.0 {
				continue;
			}
			for y in y1.min(y2).min(y3)..=y1.max(y2).max(y3) {
				for x in x1.min(x2).min(x3)..=x1.max(x2).max(x3) {
					if is_inside((x, y), ((x1, y1), (x2, y2), (x3, y3))) {
						let Some(pixel) = debug_image.get_pixel_mut_checked(x as u32, y as u32)
						else {
							continue;
						};
						let (mut r, mut g, mut b) = (pixel.0[0], pixel.0[1], pixel.0[2]);
						let edge_brightness = (relative_flow * 128.0) as u8;
						r = r.saturating_add(edge_brightness);
						// g = g.saturating_add(edge_brightness);
						// b = b.saturating_add(edge_brightness);
						*pixel = Rgb([r, g, b]);
					}
				}
			}
		}
		for (i, &(v1, v2, v3)) in triangles.iter().enumerate() {
			let (x1, y1) = vertices[v1];
			let (x2, y2) = vertices[v2];
			let (x3, y3) = vertices[v3];
			if flow_cut.contains(&i) || graph[i].iter().all(|&(_, x)| x == 0) {
				continue;
			}
			for y in y1.min(y2).min(y3)..=y1.max(y2).max(y3) {
				for x in x1.min(x2).min(x3)..=x1.max(x2).max(x3) {
					if is_inside((x, y), ((x1, y1), (x2, y2), (x3, y3))) {
						let Some(pixel) = debug_image.get_pixel_mut_checked(x as u32, y as u32)
						else {
							continue;
						};
						let (mut r, mut g, mut b) = (pixel.0[0], pixel.0[1], pixel.0[2]);
						let brightness = 200;
						// r = r.saturating_add(brightness);
						// g = g.saturating_add(brightness);
						b = b.saturating_add(brightness);
						*pixel = Rgb([r, g, b]);
					}
				}
			}
		}
		debug_image.save("factorio-map-processor-debug-5.png").unwrap();
		eprintln!("Debug images generated in {}ms", start.elapsed().as_millis());
		eprintln!("Total time: {}ms", program_start_time.elapsed().as_millis());
	};
	Ok(())
}

fn get_max_flow(graph: &Vec<Vec<(usize, Capacity)>>) -> (Vec<Capacity>, Vec<usize>) {
	let mut graph_builder: VecGraphBuilder<u32> = VecGraphBuilder::new();
	let mut edges = Vec::with_capacity(graph.len() * 3);
	let nodes = graph_builder.add_nodes(graph.len());
	let mut capacities = HashMap::new();
	for (u, vs) in graph.iter().enumerate() {
		for &(v, capacity) in vs {
			let edge = graph_builder.add_edge(nodes[u], nodes[v]);
			debug_assert!(edge.index() == edges.len());
			capacities.insert(edge, capacity);
			edges.push((u, v));
		}
	}
	debug_assert!(nodes.iter().enumerate().all(|(i, node)| i == node.index()));
	let (flow, flow_edges, minimal_cut) = maxflow::edmondskarp(
		&graph_builder.into_graph(),
		nodes[nodes.len() - 1],
		nodes[nodes.len() - 2],
		|edge| *capacities.get(&edge).unwrap(),
	);

	let mut flow = vec![0; nodes.len()];
	for (edge, edge_flow) in flow_edges {
		let (u, v) = edges[edge.index()];
		flow[u] += edge_flow;
		flow[v] += edge_flow;
	}
	(flow, minimal_cut.into_iter().map(|node| node.index()).collect())
}

fn boris_triangulation(vertices: &[(i32, i32)]) -> (Vec<(usize, usize, usize)>, Vec<Vec<usize>>) {
	// let mut thirds = HashMap::new();
	// for i in 2..vertices.len() {
	// 	let (u, v, w) = (0, i - 1, i);
	// 	thirds.insert((u, v), w);
	// 	thirds.insert((v, w), u);
	// 	thirds.insert((u, w), v);
	// }
	use spade::{
		handles::{FaceHandle, UndirectedVoronoiEdge},
		DelaunayTriangulation, Point2,
	};
	let mut triangulation: DelaunayTriangulation<Point2<f64>> = DelaunayTriangulation::new();
	for (x, y) in vertices {
		triangulation.insert(Point2::new(*x as f64, *y as f64));
	}
	let mut triangles = vec![];
	for triangle in triangulation.inner_faces() {
		let triangle: FaceHandle<_, _, _, _, _> = triangle;
		let vertices = triangle.vertices();
		let u = vertices[0].index();
		let v = vertices[1].index();
		let w = vertices[2].index();
		debug_assert!(triangles.len() == triangle.index() - 1);
		triangles.push((u, v, w));
	}
	let mut neighbors = vec![Vec::with_capacity(3); triangles.len()];
	for edge in triangulation.undirected_voronoi_edges() {
		let edge: UndirectedVoronoiEdge<_, _, _, _> = edge;
		let pair: Vec<usize> = edge
			.vertices()
			.into_iter()
			.map(|v| v.as_delaunay_face().and_then(|f| Some(f.index() - 1)).unwrap_or(usize::MAX))
			.collect();
		let u = pair[0];
		let v = pair[1];
		if u != usize::MAX && v != usize::MAX {
			neighbors[u].push(v);
			neighbors[v].push(u);
		}
	}
	(triangles, neighbors)
	// use delaunator::{triangulate, Point};
	// let points: Box<_> =
	// 	vertices.into_iter().map(|(x, y)| Point { x: *x as f64, y: *y as f64 }).collect();
	// triangulate(&points).triangles.windows(3).map(|trig| (trig[0], trig[1], trig[2])).collect()
}

#[rustfmt::skip] // Waiting for fix https://TODO
fn is_inside(point: (i32, i32), triangle: ((i32, i32), (i32, i32), (i32, i32))) -> bool {
	let (x, y) = (point.0 as isize, point.1 as isize);
	let (x1, y1) = (triangle.0.0 as isize, triangle.0.1 as isize);
	let (x2, y2) = (triangle.1.0 as isize, triangle.1.1 as isize);
	let (x3, y3) = (triangle.2.0 as isize, triangle.2.1 as isize);
	let s1 = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1);
	let s2 = (x3 - x2) * (y - y2) - (y3 - y2) * (x - x2);
	let s3 = (x1 - x3) * (y - y3) - (y1 - y3) * (x - x3);
	(s1 >= 0 && s2 >= 0 && s3 >= 0 || s1 <= 0 && s2 <= 0 && s3 <= 0)
		&& s1.abs().max(s2.abs()).max(s3.abs()) > 0
}

#[rustfmt::skip] // Waiting for fix https://TODO
fn is_inside_verbose(
	point: (usize, usize),
	triangle: ((usize, usize), (usize, usize), (usize, usize)),
) -> bool {
	let (x, y) = (point.0 as isize, point.1 as isize);
	let (x1, y1) = (triangle.0.0 as isize, triangle.0.1 as isize);
	let (x2, y2) = (triangle.1.0 as isize, triangle.1.1 as isize);
	let (x3, y3) = (triangle.2.0 as isize, triangle.2.1 as isize);
	let s1 = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1);
	let s2 = (x3 - x2) * (y - y2) - (y3 - y2) * (x - x2);
	let s3 = (x1 - x3) * (y - y3) - (y1 - y3) * (x - x3);
	eprintln!("{:?} in {:?}:", point, triangle);
	eprintln!("  {} {} {}", s1, s2, s3);
	(s1 >= 0 && s2 >= 0 && s3 >= 0 || s1 <= 0 && s2 <= 0 && s3 <= 0)
		&& s1.abs().max(s2.abs()).max(s3.abs()) > 0
}

fn remove_unreachable(map: &mut Box<[Box<[Tile]>]>) {
	let w = map.len();
	let h = map[0].len();
	let mut islands: Box<[u32]> = (0..((w * h) as u32)).collect();
	for (x, column) in map.into_iter().enumerate() {
		for (y, tile) in column.into_iter().enumerate() {
			if *tile != Tile::Water {
				if x + 1 < w {
					unite_sets(&mut islands, x + y * w, x + 1 + y * w);
				}
				if y + 1 < h {
					unite_sets(&mut islands, x + y * w, x + (y + 1) * w);
				}
			}
		}
	}
	let player_island = get_set(&mut islands, w / 2 + h / 2 * w);
	for (x, column) in map.into_iter().enumerate() {
		for (y, tile) in column.into_iter().enumerate() {
			if player_island != get_set(&mut islands, x + y * w) {
				*tile = Tile::Water;
			}
		}
	}
}

fn get_set(sets: &mut [u32], i: usize) -> usize {
	let j = sets[i] as usize;
	if i != j {
		sets[i] = get_set(sets, j) as u32;
	}
	sets[i] as usize
}

fn unite_sets(sets: &mut [u32], i: usize, j: usize) {
	sets[get_set(sets, j)] = get_set(sets, i) as u32;
}
